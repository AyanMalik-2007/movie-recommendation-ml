# src/recommend.py

import pandas as pd
import joblib
import argparse
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

# --- Аргументы командной строки ---
parser = argparse.ArgumentParser()
parser.add_argument("--user_id", type=int, required=True, help="User ID for recommendations")
parser.add_argument("--top_n", type=int, default=5, help="Number of movies to recommend")
args = parser.parse_args()

# --- Загружаем модели ---
user_item = joblib.load(MODEL_DIR / "user_item_matrix.pkl")
similarity = joblib.load(MODEL_DIR / "user_similarity.pkl")

# --- Загружаем фильмы ---
movies_file = DATA_DIR / "movies.csv"
if not movies_file.exists():
    raise FileNotFoundError(f"Place 'movies.csv' in {DATA_DIR}")

movies = pd.read_csv(
    movies_file,
    sep="\t",          # Табуляция
    header=None,
    names=["movieId", "title"],
    encoding="latin-1"
)
print("✅ Movies loaded successfully:")
print(movies.head())

# --- Рекомендации ---
user_idx = user_item.index.get_loc(args.user_id)
sim_scores = similarity[user_idx]

# Взвешенные оценки
weighted_scores = sim_scores @ user_item.values
scores = pd.Series(weighted_scores, index=user_item.columns)

# Убираем уже просмотренные фильмы
already_seen = user_item.loc[args.user_id]
scores = scores[already_seen == 0]

# Топ-N фильмов
top_movies = scores.sort_values(ascending=False).head(args.top_n)
result = movies[movies["movieId"].isin(top_movies.index)]

# --- Вывод ---
print(f"\n🎬 Recommended movies for user {args.user_id}:\n")
for _, row in result.iterrows():
    print(f"- {row['title']}")
