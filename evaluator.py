import re
import os
import pickle
import spacy
import nltk
import json
import logging
import asyncio
import PyPDF2
from asgiref.sync import async_to_sync as asgiref_async_to_sync
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import textstat
from fuzzywuzzy import fuzz, process
import numpy as np
from cachetools import TTLCache
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = nltk.corpus.stopwords.words('english')

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Action verbs for strong language detection
ACTION_VERBS = [
    "managed", "led", "developed", "created", "implemented", "designed", "built",
    "optimized", "achieved", "increased", "reduced", "improved", "launched", "streamlined",
    "spearheaded", "executed", "delivered", "transformed", "accelerated", "innovated"
]

# Cache for analysis results (TTL = 1 hour)
analysis_cache = TTLCache(maxsize=100, ttl=3600)

# Singleton ResumeAnalyzer
_resume_analyzer = None

def get_resume_analyzer():
    global _resume_analyzer
    if _resume_analyzer is None:
        _resume_analyzer = ResumeAnalyzer()
    return _resume_analyzer

class ResumeAnalyzer:
    def __init__(self, model_dir="models"):
        self.nlp = nlp
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f"
        )
        self.model_dir = model_dir
        self.load_models()
        try:
            self.skill_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # Use CPU
            )
        except (RuntimeError, ValueError):
            # Fallback to smaller model if out-of-memory or load error
            self.skill_classifier = pipeline(
                "zero-shot-classification",
                model="valhalla/distilbart-mnli-12-1",
                device=-1
            )
        logger.info("ResumeAnalyzer initialized")

    def load_models(self):
        try:
            with open(os.path.join(self.model_dir, "ats_model.pkl"), "rb") as f:
                self.ats_model = pickle.load(f)
            with open(os.path.join(self.model_dir, "skill_model.pkl"), "rb") as f:
                self.skill_model = pickle.load(f)
            with open(os.path.join(self.model_dir, "grammar_model.pkl"), "rb") as f:
                self.grammar_model = pickle.load(f)
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {str(e)}")
            raise FileNotFoundError(f"Model file not found: {str(e)}")

    def extract_text_from_file(self, file_path):
        try:
            if file_path.endswith(".pdf"):
                with open(file_path, "rb") as f:
                    pdf = PyPDF2.PdfReader(f)
                    return "".join(page.extract_text() or "" for page in pdf.pages)
            elif file_path.endswith((".docx", ".doc")):
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]
        return " ".join(tokens)

    async def analyze_grammar(self, text):
        errors = sum(text.lower().count(word) for word in self.grammar_model['common_errors'])
        word_count = len(text.split())
        score = max(0, 1 - (errors / (word_count + 1e-5)))
        return {'corrected_text': text, 'grammar_score': score}

    async def check_ats_compatibility(self, text, job_title, job_description=None):
      try:
          vec = self.ats_model['vectorizer'].transform([text])
          prob = self.ats_model['model'].predict_proba(vec)[0][1]
          if job_description:
              job_keywords = set(self.preprocess_text(job_description).split())
              resume_keywords = set(text.split())
              keyword_match = len(job_keywords.intersection(resume_keywords)) / (len(job_keywords) + 1e-5)
              prob = (prob + min(keyword_match, 1.0)) / 2
          logger.debug(f"ATS score: {prob}")
          return min(max(prob, 0.0), 1.0)
      except Exception as e:
          logger.error(f"Error in ATS compatibility: {str(e)}")
          return 0.5  # Fallback score

    async def extract_skills(self, text):
        candidate_skills = [
            "project management", "communication", "leadership", "marketing", "finance",
            "data analysis", "customer service", "sales", "writing", "design",
            "python", "sql", "javascript", "aws", "docker"
        ]
        extracted_skills = []
        for skill in candidate_skills:
            score = fuzz.partial_ratio(skill, text.lower())
            if score > 80:
                extracted_skills.append(skill)
        logger.debug(f"Extracted skills: {extracted_skills}")
        return list(set(extracted_skills))

    async def analyze_structure(self, text):
        sections = {
            'contact_info': bool(re.search(r'(name|phone|email|address|linkedin)', text, re.I)),
            'summary': bool(re.search(r'summary|profile|objective', text, re.I)),
            'experience': bool(re.search(r'experience|employment|projects|work history', text, re.I)),
            'education': bool(re.search(r'education|degree|academic background', text, re.I)),
            'skills': bool(re.search(r'skills|technical skills|competencies', text, re.I))
        }
        structure_score = sum(sections.values()) / len(sections)
        section_lengths = self.analyze_section_lengths(text)
        return {
            'structure_score': structure_score,
            'missing_sections': [k for k, v in sections.items() if not v],
            'section_lengths': section_lengths
        }

    def analyze_section_lengths(self, text):
        sections = {
            'summary': re.compile(r'summary|profile|objective', re.I),
            'experience': re.compile(r'experience|employment|projects|work history', re.I),
            'education': re.compile(r'education|degree|academic background', re.I),
            'skills': re.compile(r'skills|technical skills|competencies', re.I)
        }
        lengths = {}
        for section, pattern in sections.items():
            matches = pattern.finditer(text)
            section_text = ""
            for match in matches:
                start = match.start()
                end = text.find('\n\n', start) if text.find('\n\n', start) != -1 else len(text)
                section_text += text[start:end]
            lengths[section] = len(section_text.split())
        return lengths

    async def analyze_formatting(self, text):
        bullets = len(re.findall(r"[\*\•\-]", text))
        dates = len(re.findall(r"(19|20)\d{2}", text))
        words = len(re.findall(r'\b\w+\b', text))
        pronouns = len(re.findall(r'\b(i|my|me|mine)\b', text, re.IGNORECASE))
        readability = textstat.flesch_reading_ease(text)
        action_verb_count = sum(1 for verb in ACTION_VERBS if verb in text.lower())

        score = 0
        if 150 <= words <= 600:
            score += 30
        elif 100 <= words <= 800:
            score += 20
        if pronouns <= 2:  # Relaxed pronoun penalty
            score += 15
        score += min(20, bullets * 2 + dates * 2)
        if 50 <= readability <= 90:  # Wider readability range
            score += 15
        score += min(20, action_verb_count * 4)

        score = max(min(score, 100), 0)
        logger.debug(f"Formatting score components: words={words}, pronouns={pronouns}, "
                    f"bullets={bullets}, dates={dates}, readability={readability}, "
                    f"action_verbs={action_verb_count}, total={score}")
        return score

    async def analyze_entities(self, text):
        doc = self.nlp(text)
        entities = {
            'persons': [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],
            'organizations': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
            'locations': [ent.text for ent in doc.ents if ent.label_ == 'GPE']
        }
        return entities

    async def analyze_sentiment(self, text):
        result = self.sentiment_analyzer(text[:512])[0]
        return {'sentiment': result['label'], 'score': result['score']}

    async def generate_feedback(self, analysis_results, job_title=None, job_description=None):
      strengths = []
      suggestions = []
      grammar_score = analysis_results['grammar']['grammar_score']
      ats_score = analysis_results['ats_score']
      structure = analysis_results['structure']
      formatting_score = analysis_results['formatting_score']
      skills = analysis_results['skills']
      entities = analysis_results['entities']
      sentiment = analysis_results['sentiment']

      # Grammar feedback
      if grammar_score >= 0.9:
          strengths.append("Excellent grammar and spelling.")
      else:
          suggestions.append("Improve grammar and reduce spelling errors for a more professional resume.")

      # ATS feedback (general-purpose)
      if ats_score >= 0.8:
          strengths.append("Resume is well-optimized for ATS systems.")
      else:
          suggestions.append("Incorporate more relevant keywords to enhance ATS compatibility.")

      # Structure feedback
      if len(structure['missing_sections']) == 0:
          strengths.append("All key resume sections are present.")
      else:
          suggestions.append(f"Add missing sections: {', '.join(structure['missing_sections'])}.")

      # Skills feedback (generalized)
      if len(skills) >= 3:
          strengths.append("Good range of skills listed.")
      else:
          suggestions.append("Highlight additional relevant skills to strengthen the resume.")

      # Formatting feedback
      if formatting_score >= 70:
          strengths.append("Effective formatting enhances readability.")
      else:
          suggestions.append("Use consistent formatting, bullets, and avoid personal pronouns.")

      # Sentiment feedback
      if sentiment['sentiment'] == 'POSITIVE' and sentiment['score'] > 0.7:
          strengths.append("Resume conveys a positive and confident tone.")
      else:
          suggestions.append("Use more positive and action-oriented language.")

      # Entity feedback
      if entities['persons']:
          suggestions.append("Remove personal names from experience/education to maintain anonymity.")
      if len(entities['organizations']) > 0:
          strengths.append("Relevant organizations mentioned, adding credibility.")

      # Role-specific feedback
      if job_title and job_description:
          job_doc = self.nlp(self.preprocess_text(job_description))
          resume_doc = self.nlp(analysis_results['clean_text'])
          similarity = resume_doc.similarity(job_doc)
          if similarity > 0.8:
              strengths.append(f"Strong alignment with {job_title} requirements.")
          else:
              suggestions.append(f"Tailor resume to better match {job_title} requirements.")

      return strengths, suggestions

    async def analyze_resume(self, file_path, job_title=None, job_description=None, skip_ats=False):
        import time
        start_time = time.time()
        cache_key = f"{file_path}_{job_title}_{job_description}_{skip_ats}"
        if cache_key in analysis_cache:
            logger.info(f"Returning cached result for {file_path}")
            return analysis_cache[cache_key]

        try:
            text = self.extract_text_from_file(file_path)
            clean_text = self.preprocess_text(text)

            if len(clean_text.split()) < 30:
                result = {
                    'error': "Text too short to be a valid resume.",
                    'strengths': [],
                    'suggestions': ["This document appears too brief to be a resume."],
                    'weaknesses': ["Resume lacks sufficient content or structure."],
                    'finalscore': 0,
                    'match_level': 'Not a Resume',
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
                analysis_cache[cache_key] = result
                return result

            # Run analyses concurrently
            grammar, skills, structure, formatting_score, entities, sentiment = await asyncio.gather(
                self.analyze_grammar(text),
                self.extract_skills(clean_text),
                self.analyze_structure(text),
                self.analyze_formatting(text),
                self.analyze_entities(text),
                self.analyze_sentiment(text)
            )

            ats_score = 0.0
            if not skip_ats:
                ats_score = await self.check_ats_compatibility(clean_text, job_title, job_description)

            # Store clean_text for role-specific feedback
            analysis_results = {
                'grammar': grammar,
                'ats_score': ats_score,
                'structure': structure,
                'skills': skills,
                'formatting_score': formatting_score,
                'entities': entities,
                'sentiment': sentiment,
                'clean_text': clean_text
            }

            strengths, suggestions = await self.generate_feedback(analysis_results, job_title, job_description)

            # Calculate normalized score
            finalscore = int(
                (min(ats_score, 1.0) * 40 if not skip_ats else 0) +
                min(structure['structure_score'], 1.0) * 30 +
                min(formatting_score / 100.0, 1.0) * 20 +
                (min(sentiment['score'], 1.0) if sentiment['sentiment'] == 'POSITIVE' else 0) * 10
            )
            finalscore = min(finalscore, 100)  # Cap at 100
            logger.debug(f"Score components: ats={ats_score}, structure={structure['structure_score']}, "
                        f"formatting={formatting_score}, sentiment={sentiment['score']}, final={finalscore}")

            match_level = (
                "Excellent" if finalscore >= 90 else
                "Good" if finalscore >= 75 else
                "Fair" if finalscore >= 50 else
                "Poor"
            )

            result = {
                'grammar': grammar,
                'ats_score': ats_score,
                'skills': skills,
                'structure': structure,
                'formatting_score': formatting_score,
                'entities': entities,
                'sentiment': sentiment,
                'word_count': len(clean_text.split()),
                'strengths': strengths,
                'suggestions': suggestions,
                'weaknesses': suggestions,
                'finalscore': finalscore,
                'match_level': match_level,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

            analysis_cache[cache_key] = result
            logger.info(f"Analysis completed for {file_path} in {time.time() - start_time:.2f} seconds")
            return result

        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}: {str(e)}")
            result = {
                'error': f"File not found: {file_path}",
                'strengths': [],
                'suggestions': ["Ensure the resume file exists and the path is correct."],
                'weaknesses': ["File processing failed."],
                'finalscore': 0,
                'match_level': 'Error',
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            analysis_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Error analyzing resume {file_path}: {str(e)}")
            result = {
                'error': str(e),
                'strengths': [],
                'suggestions': ["An unexpected error occurred. Please try again with a valid document."],
                'weaknesses': ["File processing failed."],
                'finalscore': 0,
                'match_level': 'Error',
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            analysis_cache[cache_key] = result
            return result

async def evaluate_resume(file_path: str, job_title: str = None , job_description: str = None, user_id: str = None, use_custom_keywords: int = 0, custom_keywords: str = "") -> dict:
    analyzer = ResumeAnalyzer()
    if use_custom_keywords:
        # Use custom keywords in analysis process
        # Override job_description with custom_keywords for ATS check
        job_description = custom_keywords
        # Skip ATS model scoring when using custom keywords
        return await analyzer.analyze_resume(file_path, job_title, job_description, skip_ats=True)
    else:
        return await analyzer.analyze_resume(file_path, job_title, job_description)

