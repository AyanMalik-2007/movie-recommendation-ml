
# Movie Recommendation ML

A **user-based collaborative filtering movie recommendation system** built with Python and machine learning.

## Project Overview

This project implements a movie recommendation system using user-based collaborative filtering. It predicts movies a user is likely to enjoy based on the ratings of similar users.

## Project Structure


```markdown
movie-recommendation-ml/
├── data/              # CSV files: ratings.csv, movies.csv
├── model/             # Trained models saved here
├── src/
│   ├── train.py       # Script to train the model
│   └── recommend.py   # Script to get movie recommendations
├── .gitignore
├── README.md
└── requirements.txt

````

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-ml.git
cd movie-recommendation-ml
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your `ratings.csv` and `movies.csv` files into the `data/` folder.

---

## Usage

### 1. Train the model

```bash
python3 src/train.py
```

### 2. Get movie recommendations

```bash
python3 src/recommend.py --user_id 3 --top_n 5
```

Replace `3` with your user ID and `5` with the number of recommendations.

---

## Notes

* The `data/` folder contains user ratings and movie information.
* The `model/` folder stores the trained user-item matrix and similarity matrix.
* CSV file formats:

  * `ratings.csv` — tab-separated, columns: `userId`, `movieId`, `rating`, `timestamp`
  * `movies.csv` — tab-separated, columns: `movieId`, `title`


