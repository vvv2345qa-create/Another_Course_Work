import argparse
from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_bundle(path: str):
    bundle = joblib.load(path)
    required_keys = {"embedding_model_name", "classifier", "label_encoder"}
    missing = required_keys - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    return bundle


def predict_texts(bundle, texts: list[str], threshold: float) -> pd.DataFrame:
    embedding_model = SentenceTransformer(bundle["embedding_model_name"])
    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    classifier = bundle["classifier"]
    label_encoder = bundle["label_encoder"]

    probabilities = classifier.predict_proba(embeddings)
    class_names = label_encoder.classes_

    rows = []
    for text, row in zip(texts, probabilities):
        best_index = int(row.argmax())
        best_label = class_names[best_index]
        best_score = float(row[best_index])

        decision = "block" if best_label != "benign" and best_score >= threshold else "allow"
        rows.append(
            {
                "text": text,
                "predicted_label": best_label,
                "risk_score": round(best_score, 4),
                "decision": decision,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_data_dir = Path.home() / "Desktop" / "llm_guard_dataset"
    parser.add_argument("--model", default="models/guard_model.joblib")
    parser.add_argument("--csv", default=str(default_data_dir / "test.csv"))
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    bundle = load_bundle(args.model)
    df = pd.read_csv(args.csv)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    result = predict_texts(bundle, df["text"].astype(str).tolist(), args.threshold)

    if "label" in df.columns:
        result.insert(1, "true_label", df["label"].astype(str).tolist())

    print(result.to_string(index=False))

    if args.out:
        result.to_csv(args.out, index=False, encoding="utf-8")
        print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
