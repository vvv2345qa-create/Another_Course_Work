import argparse
from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


def load_dataset(path: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    return df["text"].astype(str).tolist(), df["label"].astype(str).tolist()


def encode_texts(model: SentenceTransformer, texts: list[str]):
    return model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def train_classifier(embeddings, labels: list[str]):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    classifier.fit(embeddings, y)
    return classifier, label_encoder


def save_bundle(
    out_path: str,
    embedding_model_name: str,
    classifier: LogisticRegression,
    label_encoder: LabelEncoder,
) -> None:
    bundle = {
        "embedding_model_name": embedding_model_name,
        "classifier": classifier,
        "label_encoder": label_encoder,
    }
    joblib.dump(bundle, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_data_dir = Path.home() / "Desktop" / "llm_guard_dataset"
    parser.add_argument("--train-csv", default=str(default_data_dir / "train.csv"))
    parser.add_argument("--test-csv", default=str(default_data_dir / "test.csv"))
    parser.add_argument("--out", default="models/guard_model.joblib")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    train_texts, train_labels = load_dataset(args.train_csv)
    test_texts, test_labels = load_dataset(args.test_csv)

    embedding_model = SentenceTransformer(args.embedding_model)

    x_train = encode_texts(embedding_model, train_texts)
    x_test = encode_texts(embedding_model, test_texts)

    classifier, label_encoder = train_classifier(x_train, train_labels)

    y_test = label_encoder.transform(test_labels)
    y_pred = classifier.predict(x_test)

    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    save_bundle(args.out, args.embedding_model, classifier, label_encoder)
    print(f"Saved model bundle to {args.out}")


if __name__ == "__main__":
    main()
