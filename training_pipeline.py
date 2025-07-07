# training_pipeline.py
import pandas as pd
import numpy as np
import os
import pickle
import random
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pdfminer.high_level import extract_text
from docx import Document


class ResumeAnalysisTrainer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def extract_text_from_file(self, file_path):
        if file_path.endswith(".pdf"):
            return extract_text(file_path)
        elif file_path.endswith((".docx", ".doc")):
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            return ""

    def load_dataset(self, folder_path):
        data = []
        for label_name, label_score in [
            ("good", 0.9),
            ("bad", 0.6),
            ("not_resume", 0.3),
        ]:
            category_path = os.path.join(folder_path, label_name)
            if not os.path.isdir(category_path):
                continue
            for fname in os.listdir(category_path):
                if fname.lower().endswith((".pdf", ".doc", ".docx")):
                    full_path = os.path.join(category_path, fname)
                    text = self.extract_text_from_file(full_path)
                    if len(text) > 100:
                        skills = []
                        if "python" in text.lower():
                            skills.append("python")
                        if "java" in text.lower():
                            skills.append("java")
                        if not skills:
                            skills = ["excel"]
                        data.append(
                            {
                                "text": text,
                                "ats_score": label_score,
                                "grammar_errors": random.randint(1, 7),
                                "skills": skills,
                            }
                        )
        return pd.DataFrame(data)

    def train_ats_model(self, data):
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(data["text"])
        y = (data["ats_score"] > 0.75).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("\nATS Model Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

        return {"model": model, "vectorizer": vectorizer}

    def train_skill_extractor(self, data):
        all_skills = list(
            set([skill for sublist in data["skills"] for skill in sublist])
        )
        skill_features = pd.DataFrame()
        for skill in all_skills:
            skill_features[skill] = (
                data["text"].str.contains(skill, case=False).astype(int)
            )

        skill_models = {}
        vectorizer = TfidfVectorizer(max_features=2000)
        X = vectorizer.fit_transform(data["text"])

        for skill in all_skills:
            y = skill_features[skill]
            if y.sum() > 1:
                model = RandomForestClassifier(n_estimators=50)
                model.fit(X, y)
                skill_models[skill] = model

        return {
            "models": skill_models,
            "vectorizer": vectorizer,
            "skill_vocab": all_skills,
        }

    def train_grammar_model(self, data):
        return {"common_errors": ["their", "there", "your", "you're"]}

    def save_models(self, models, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/ats_model.pkl", "wb") as f:
            pickle.dump(models["ats_model"], f)
        with open(f"{output_dir}/skill_model.pkl", "wb") as f:
            pickle.dump(models["skill_model"], f)
        with open(f"{output_dir}/grammar_model.pkl", "wb") as f:
            pickle.dump(models["grammar_model"], f)

    def run_training(self):
        print("Loading training data...")
        data = self.load_dataset(
            r"C:\\Users\\Mega Store\\OneDrive\\CV MANAGIZER\\train\\data_set"
        )

        print("Training ATS compatibility model...")
        ats_model = self.train_ats_model(data)

        print("Training skill extractor...")
        skill_model = self.train_skill_extractor(data)

        print("Training grammar model...")
        grammar_model = self.train_grammar_model(data)

        models = {
            "ats_model": ats_model,
            "skill_model": skill_model,
            "grammar_model": grammar_model,
        }

        print("Saving models...")
        self.save_models(models, "models")

        print("\n✅ Training complete!")


if __name__ == "__main__":
    trainer = ResumeAnalysisTrainer()
    trainer.run_training()
