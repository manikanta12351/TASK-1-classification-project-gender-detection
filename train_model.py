from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/names_gender.csv")
MODEL_PATH = Path("models/name_gender_model.joblib")


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    names: list[str] = []
    genders: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            name = (row.get("name") or "").strip()
            gender = (row.get("gender") or "").strip().lower()
            if name and gender in {"male", "female"}:
                names.append(name)
                genders.append(gender)

    return names, genders


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(2, 4))),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run generate_dataset.py first."
        )

    names, genders = load_dataset(DATA_PATH)
    if len(names) < 10:
        raise ValueError("Dataset is too small to train a classifier.")

    x_train, x_test, y_train, y_test = train_test_split(
        names,
        genders,
        test_size=0.2,
        random_state=42,
        stratify=genders,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
