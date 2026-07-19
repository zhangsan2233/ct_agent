import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CT-RATE report similarity index.")
    parser.add_argument("--case-index", default="artifacts/prepared/case_index.csv")
    parser.add_argument(
        "--out",
        default="artifacts/prepared/similar_case_index.joblib",
    )
    parser.add_argument("--max-features", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.case_index)
    frame = pd.read_csv(source_path)
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(frame["report_text"].fillna(""))
    source_stat = source_path.stat()
    artifact = {
        "version": 2,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)
    print(f"Wrote {matrix.shape[0]} x {matrix.shape[1]} index to {output_path}")


if __name__ == "__main__":
    main()
