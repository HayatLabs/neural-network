from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class FeatureExtractor:
    def __init__(self):
      
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), 
            max_features=5000,
            sublinear_tf=True
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def save_vectorizer(self, path):
        joblib.dump(self.vectorizer, path)

    def load_vectorizer(self, path):
        self.vectorizer = joblib.load(path)