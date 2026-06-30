# Movie Genre Predictor

Predicts a movie's genre from its plot summary using classical NLP techniques —
TF-IDF feature extraction combined with three scikit-learn classifiers.

## Project structure

```
movie_genre_predictor/
├── README.md
├── requirements.txt
├── dataset.py      # Labelled plot-summary dataset & DataFrame loader
├── model.py        # TF-IDF pipelines, training/evaluation, inference
└── main.py         # CLI entry point (train / interactive predict)
```

## Classifiers compared

| Model | Technique | Notes |
|---|---|---|
| Naive Bayes | MultinomialNB | Fast probabilistic baseline |
| Logistic Regression | L-BFGS, C=5 | Best cross-validated F1 |
| SVM | LinearSVC | Highest single-run test F1 |

All three share the same TF-IDF front-end: bigrams, 10 000 features,
sublinear TF scaling, English stop-words removed.

## Genres

Action · Animation · Comedy · Drama · Horror · Romance · Sci-Fi · Thriller

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-username/movie_genre_predictor.git
cd movie_genre_predictor
pip install -r requirements.txt

# 2. Train all three models and print a comparison report
python main.py

# 3. Interactive prediction from the command line
python main.py --predict

# 4. Use the predictor as a library
python - <<'EOF'
from model import predict_genre
result = predict_genre("A detective hunts a serial killer who leaves cryptic puzzles.")
print(result["genre"])       # Thriller
print(result["top3"])        # [(genre, pct), ...]
EOF
```

## Results (320 samples, 80/20 split)

| Model | Test F1 | CV F1 (5-fold) |
|---|---|---|
| Naive Bayes | 39.5 % | 39.5 % ± 7.4 % |
| Logistic Regression | 42.1 % | **42.7 % ± 6.2 %** |
| SVM (LinearSVC) | 43.3 % | 41.7 % ± 6.2 % |

The best cross-validated model is saved automatically to `movie_genre_model.pkl`.

## Improving accuracy

- **Larger dataset** — replace the built-in plots with the
  [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/) (42 k plots).
- **Word embeddings** — swap TF-IDF for `sentence-transformers`
  (`all-MiniLM-L6-v2`) for richer semantic features.
- **Fine-tuned transformer** — a DistilBERT classifier typically reaches
  80–90 % F1 on genre classification tasks.

## License

MIT
