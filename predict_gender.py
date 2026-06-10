from __future__ import annotations

import sys
from pathlib import Path

import joblib

MODEL_PATH = Path("models/name_gender_model.joblib")


def predict_name(name: str) -> str:
    model = joblib.load(MODEL_PATH)
    return str(model.predict([name])[0])


def main(argv: list[str]) -> int:
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Train it first.")
        return 1

    if len(argv) > 1:
        name = " ".join(argv[1:]).strip()
    else:
        name = input("Enter a name: ").strip()

    if not name:
        print("Please provide a name.")
        return 1

    prediction = predict_name(name)
    print(f"{name} -> {prediction}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
