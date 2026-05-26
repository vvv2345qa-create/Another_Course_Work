import argparse
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score


matplotlib.use("Agg")


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df[["text", "label"]].copy()


def predict_dataframe(bundle: dict, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    embedding_model = SentenceTransformer(bundle["embedding_model_name"])
    embeddings = embedding_model.encode(
        df["text"].astype(str).tolist(),
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    classifier = bundle["classifier"]
    label_encoder = bundle["label_encoder"]

    probabilities = classifier.predict_proba(embeddings)
    predicted_ids = classifier.predict(embeddings)
    predicted_labels = label_encoder.inverse_transform(predicted_ids)
    true_ids = label_encoder.transform(df["label"].astype(str).tolist())

    result = df.copy()
    result.insert(0, "split", split_name)
    result["true_label_id"] = true_ids
    result["predicted_label"] = predicted_labels
    result["predicted_label_id"] = predicted_ids
    result["confidence"] = probabilities.max(axis=1).round(4)
    result["is_correct"] = result["label"] == result["predicted_label"]
    return result


def save_label_mapping(bundle: dict, out_dir: Path) -> None:
    label_encoder = bundle["label_encoder"]
    mapping = pd.DataFrame(
        {
            "label_id": range(len(label_encoder.classes_)),
            "label": label_encoder.classes_,
        }
    )
    mapping.to_csv(out_dir / "label_mapping.csv", index=False, encoding="utf-8")


def plot_accuracy(train_accuracy: float, test_accuracy: float, out_dir: Path) -> None:
    labels = ["Train", "Test"]
    values = [train_accuracy * 100, test_accuracy * 100]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=["#3b82f6", "#16a34a"])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy, %")
    plt.title("Model Accuracy by Dataset Split")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_train_test.png", dpi=160)
    plt.close()


def plot_predicted_label_distribution(results: pd.DataFrame, out_dir: Path) -> None:
    distribution = (
        results["predicted_label"]
        .value_counts(normalize=True)
        .mul(100)
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    bars = plt.bar(distribution.index, distribution.values, color="#f97316")
    plt.ylabel("Detected labels, %")
    plt.title("Predicted Label Distribution")
    plt.xticks(rotation=25, ha="right")

    for bar, value in zip(bars, distribution.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_dir / "predicted_label_distribution.png", dpi=160)
    plt.close()


def main() -> None:
    default_data_dir = Path.home() / "Desktop" / "llm_guard_dataset"
    default_out_dir = Path.home() / "Desktop" / "llm_guard_results"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/guard_model.joblib")
    parser.add_argument("--train-csv", default=str(default_data_dir / "train.csv"))
    parser.add_argument("--test-csv", default=str(default_data_dir / "test.csv"))
    parser.add_argument("--out-dir", default=str(default_out_dir))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model)

    train_df = load_dataset(args.train_csv)
    test_df = load_dataset(args.test_csv)

    train_results = predict_dataframe(bundle, train_df, "train")
    test_results = predict_dataframe(bundle, test_df, "test")
    all_results = pd.concat([train_results, test_results], ignore_index=True)

    train_accuracy = accuracy_score(train_results["label"], train_results["predicted_label"])
    test_accuracy = accuracy_score(test_results["label"], test_results["predicted_label"])

    train_results.to_csv(out_dir / "train_predictions.csv", index=False, encoding="utf-8")
    test_results.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8")
    all_results.to_csv(out_dir / "all_predictions.csv", index=False, encoding="utf-8")

    pd.DataFrame(
        [
            {"split": "train", "accuracy": train_accuracy, "accuracy_percent": train_accuracy * 100},
            {"split": "test", "accuracy": test_accuracy, "accuracy_percent": test_accuracy * 100},
        ]
    ).to_csv(out_dir / "accuracy_summary.csv", index=False, encoding="utf-8")

    save_label_mapping(bundle, out_dir)
    plot_accuracy(train_accuracy, test_accuracy, out_dir)
    plot_predicted_label_distribution(all_results, out_dir)

    print(f"Train accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Saved evaluation files to {out_dir}")


if __name__ == "__main__":
    main()
