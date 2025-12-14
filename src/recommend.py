# src/recommend.py

import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

similarity = joblib.load(MODEL_DIR / "user_similarity.pkl")
user_movie_matrix = joblib.load(MODEL_DIR / "user_movie_matrix.pkl")

def recommend(user_id, top_n=3):
    idx = user_movie_matrix.index.tolist().index(user_id)
    scores = similarity[idx]
    similar_users = scores.argsort()[::-1][1:]

    recommendations = user_movie_matrix.iloc[similar_users].mean()
    recommendations = recommendations[user_movie_matrix.loc[user_id] == 0]

    return recommendations.sort_values(ascending=False).head(top_n)

if __name__ == "__main__":
    print(recommend(1))
