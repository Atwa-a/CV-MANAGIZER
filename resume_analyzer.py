import re
import pandas as pd
import numpy as np
import spacy
import nltk
from pdfminer.high_level import extract_text
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize NLP components
nlp = spacy.load("en_core_web_lg")
nltk.download("stopwords")
stop_words = nltk.corpus.stopwords.words("english")


class ResumeAnalyzer:
    def __init__(self):
        # Initialize models
        self.grammar_checker = pipeline(
            "text2text-generation", model="vennify/t5-base-grammar-correction"
        )
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.keyword_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased"
        )
        self.keyword_tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

        # Load trained models (would be loaded from disk in production)
        self.ats_model = self._train_ats_model()  # Mock function
        self.skill_extractor = self._train_skill_extractor()  # Mock function

    def _train_ats_model(self):
        """Train an ATS compatibility classifier (simplified example)"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # In production, you'd load a real dataset of resumes with ATS scores
        data = pd.DataFrame(
            {
                "text": [
                    "software engineer python java",
                    "marketing manager",
                    "data scientist",
                ],
                "ats_score": [0.92, 0.85, 0.88],
            }
        )

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data["text"])
        y = data["ats_score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model

    def _train_skill_extractor(self):
        """Train a skill extraction model (simplified example)"""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB

        # In production, use a dataset of resumes with labeled skills
        data = pd.DataFrame(
            {
                "text": [
                    "experienced in python and machine learning",
                    "proficient in java and spring boot",
                    "skilled in data analysis and SQL",
                ],
                "skills": [
                    ["python", "machine learning"],
                    ["java", "spring boot"],
                    ["data analysis", "SQL"],
                ],
            }
        )

        vectorizer = CountVectorizer(ngram_range=(1, 2))
        X = vectorizer.fit_transform(data["text"])

        # Multi-label classification setup would be more complex in reality
        model = MultinomialNB()
        model.fit(X, [1, 1, 1])  # Simplified for example
        return model

    def extract_text_from_file(self, file_path):
        """Extract text from PDF or DOCX files"""
        if file_path.endswith(".pdf"):
            return extract_text(file_path)
        elif file_path.endswith((".docx", ".doc")):
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            raise ValueError("Unsupported file format")

    def preprocess_text(self, text):
        """Clean and preprocess resume text"""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
        return text

    def analyze_grammar(self, text):
        """Check grammar and spelling"""
        results = self.grammar_checker(text)
        return {
            "corrected_text": results[0]["generated_text"],
            "grammar_score": len(text.split())
            / (len(text.split()) + len(results[0]["generated_text"].split())),
        }

    def check_ats_compatibility(self, text, job_title):
        """Check how well the resume matches ATS requirements"""
        processed_text = self.preprocess_text(text)
        features = self.ats_model.named_steps["tfidf"].transform([processed_text])
        score = self.ats_model.predict_proba(features)[0][1]
        return score

    def extract_skills(self, text):
        """Extract technical skills from resume"""
        doc = nlp(text)
        skills = []

        # Rule-based extraction
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                skills.append(ent.text)

        # Pattern matching
        for chunk in doc.noun_chunks:
            if any(
                token.text in ["experience", "proficient", "skilled"]
                for token in chunk.root.head.children
            ):
                skills.append(chunk.text)

        return list(set(skills))

    def analyze_structure(self, text):
        """Evaluate resume structure quality"""
        sections = {
            "contact_info": bool(re.search(r"(name|phone|email|address)", text, re.I)),
            "summary": bool(re.search(r"summary|profile", text, re.I)),
            "experience": bool(re.search(r"experience|work history", text, re.I)),
            "education": bool(re.search(r"education|degree", text, re.I)),
            "skills": bool(re.search(r"skills|technical skills", text, re.I)),
        }

        structure_score = sum(sections.values()) / len(sections)
        return {
            "structure_score": structure_score,
            "missing_sections": [k for k, v in sections.items() if not v],
        }

    def generate_feedback(self, analysis_results):
        """Generate comprehensive feedback based on analysis"""
        feedback = []

        if analysis_results["grammar"]["grammar_score"] < 0.9:
            feedback.append(
                "Grammar and spelling could be improved. Consider reviewing with a tool like Grammarly."
            )

        if analysis_results["ats_score"] < 0.8:
            feedback.append(
                f"ATS compatibility score is {analysis_results['ats_score']*100:.0f}%. Consider adding more keywords relevant to your target job."
            )

        if len(analysis_results["structure"]["missing_sections"]) > 0:
            feedback.append(
                f"Missing sections: {', '.join(analysis_results['structure']['missing_sections'])}. These are important for a complete resume."
            )

        if len(analysis_results["skills"]) < 5:
            feedback.append(
                "Consider adding more technical skills to better showcase your qualifications."
            )

        return feedback

    def analyze_resume(self, file_path, job_title="Software Engineer"):
        """Main analysis pipeline"""
        try:
            # Extract and preprocess text
            text = self.extract_text_from_file(file_path)
            clean_text = self.preprocess_text(text)

            # Run all analyses
            grammar = self.analyze_grammar(text)
            ats_score = self.check_ats_compatibility(clean_text, job_title)
            skills = self.extract_skills(text)
            structure = self.analyze_structure(text)

            # Compile results
            results = {
                "grammar": grammar,
                "ats_score": ats_score,
                "skills": skills,
                "structure": structure,
                "word_count": len(clean_text.split()),
                "feedback": self.generate_feedback(
                    {
                        "grammar": grammar,
                        "ats_score": ats_score,
                        "structure": structure,
                        "skills": skills,
                    }
                ),
            }

            return results

        except Exception as e:
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    analyzer = ResumeAnalyzer()
    sample_results = analyzer.analyze_resume("sample_resume.pdf")
    print("Analysis Results:")
    print(f"Grammar Score: {sample_results['grammar']['grammar_score']:.0%}")
    print(f"ATS Compatibility: {sample_results['ats_score']:.0%}")
    print(f"Key Skills: {', '.join(sample_results['skills'][:5])}")
    print("\nFeedback:")
    for item in sample_results["feedback"]:
        print(f"- {item}")
