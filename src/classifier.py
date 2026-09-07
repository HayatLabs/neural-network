# src/classifier.py
from sklearn.linear_model import LogisticRegression
from src.feature_engineering import FeatureExtractor
import joblib
import os

class IntentClassifier:
    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver='liblinear', multi_class='ovr')
        self.fe = FeatureExtractor()
        self.model_dir = "models"
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def train(self, texts, labels):
        X = self.fe.fit_transform(texts)
        self.model.fit(X, labels)
        joblib.dump(self.model, os.path.join(self.model_dir, 'intent_model.pkl'))
        self.fe.save_vectorizer(os.path.join(self.model_dir, 'vectorizer.pkl'))
        print("Training complete and models saved.")

    def predict_intent(self, text):
        loaded_model = joblib.load(os.path.join(self.model_dir, 'intent_model.pkl'))
        self.fe.load_vectorizer(os.path.join(self.model_dir, 'vectorizer.pkl'))
        
        vectorized_input = self.fe.transform([text])
        prediction = loaded_model.predict(vectorized_input)[0]
        confidence = max(loaded_model.predict_proba(vectorized_input)[0])
        
        return {
            "intent": prediction,
            "confidence": f"{confidence * 100:.2f}%"
        }