import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEncoder:
    def __init__(self):
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.text_vectorizer = TfidfVectorizer(max_features=100)

    def fit(self, df):
        self.cat_encoder.fit(df[["category"]])
        self.scaler.fit(df[["price"]])
        self.text_vectorizer.fit(df["description"])

    def transform(self, df):
        cat_features = self.cat_encoder.transform(df[["category"]]).toarray()
        num_features = self.scaler.transform(df[["price"]])
        text_features = self.text_vectorizer.transform(df["description"]).toarray()

        return np.hstack([cat_features, num_features, text_features])
