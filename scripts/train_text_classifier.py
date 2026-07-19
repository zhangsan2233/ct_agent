import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + LogisticRegression report baseline.")
    parser.add_argument("--case-index", default="artifacts/prepared/case_index.csv")
    parser.add_argument("--out", default="artifacts/text_classifier.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.case_index)
    labels = sorted({label for labels in df["labels"].fillna("") for label in labels.split(";") if label})
    if not labels:
        raise ValueError("No labels found in case index.")
    y = pd.DataFrame({label: df["labels"].fillna("").str.contains(label, regex=False).astype(int) for label in labels})
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000))),
        ]
    )
    pipeline.fit(df["report_text"].fillna(""), y)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipeline, "labels": labels}, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
