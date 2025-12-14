# src/train.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import os

# Пути
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

def train():
    # --- 1. Загружаем оценки ---
    ratings_file = DATA_DIR / "ratings.csv"
    if not ratings_file.exists():
        raise FileNotFoundError(f"Place 'ratings.csv' in {DATA_DIR}")

    ratings = pd.read_csv(
        ratings_file,
        sep="\t",           # Табуляция
        header=None,        # Нет заголовка
        names=["userId", "movieId", "rating", "timestamp"]
    )
    ratings = ratings.drop("timestamp", axis=1)
    print("✅ Ratings loaded successfully:")
    print(ratings.head())

    # --- 2. User-Item матрица ---
    user_item_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    # --- 3. Косинусная схожесть пользователей ---
    similarity = cosine_similarity(user_item_matrix)

    # --- 4. Сохраняем модели ---
    joblib.dump(user_item_matrix, MODEL_DIR / "user_item_matrix.pkl")
    joblib.dump(similarity, MODEL_DIR / "user_similarity.pkl")
    print(f"\n✅ Model trained and saved successfully in '{MODEL_DIR}'")

if __name__ == "__main__":
    train()
