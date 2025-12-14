# src/train.py

import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ratings.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

def train():
    df = pd.read_csv(DATA_PATH)

    user_movie_matrix = df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    similarity = cosine_similarity(user_movie_matrix)

    joblib.dump(similarity, MODEL_DIR / "user_similarity.pkl")
    joblib.dump(user_movie_matrix, MODEL_DIR / "user_movie_matrix.pkl")

    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train()
