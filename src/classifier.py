# src/classifier.py
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.feature_engineering import FeatureExtractor

class IntentClassifier:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, "intent_model.pkl")
        self.vec_path = os.path.join(self.model_dir, "vectorizer.pkl")
        
        self.fe = FeatureExtractor()
        self.model = LogisticRegression(C=1000.0, solver='liblinear', multi_class='ovr')
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def train(self, texts, labels):

        print("Training: Extracting features...")
        X = self.fe.fit_transform(texts)
        
        print("Training: Fitting Logistic Regression model...")
        self.model.fit(X, labels)
        
        joblib.dump(self.model, self.model_path)
        self.fe.save_vectorizer(self.vec_path)
        print(f"Model trained and saved in {self.model_dir}/")

    def predict(self, text):
   
        if not os.path.exists(self.model_path):
            raise Exception("Model files not found! Please train the model first.")

        loaded_model = joblib.load(self.model_path)
        self.fe.load_vectorizer(self.vec_path)
        
        X_input = self.fe.transform([text])
        
        intent = loaded_model.predict(X_input)[0]
        
        probs = loaded_model.predict_proba(X_input)[0]
        confidence = np.max(probs)
        
        return {
            "intent": intent,
            "confidence": f"{confidence * 100:.2f}%"
        }