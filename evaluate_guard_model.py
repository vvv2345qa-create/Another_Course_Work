import argparse
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)


matplotlib.use("Agg")


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df[["text", "label"]].copy()


def load_test_texts(path: str) -> pd.DataFrame:
    test_path = Path(path)
    if test_path.suffix.lower() == ".csv":
        df = pd.read_csv(test_path)
        if "text" not in df.columns:
            raise ValueError("Test CSV must contain a 'text' column")
        return df[["text"]].copy()

    texts = [
        line.strip()
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame({"text": texts})


def load_expected_labels(path: str) -> list[str]:
    labels_path = Path(path)
    if labels_path.suffix.lower() == ".csv":
        df = pd.read_csv(labels_path)
        if "label" in df.columns:
            return df["label"].astype(str).str.strip().tolist()
        if len(df.columns) == 1:
            return df.iloc[:, 0].astype(str).str.strip().tolist()
        raise ValueError("Expected labels CSV must contain a 'label' column or exactly one column")

    return [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def attach_expected_labels(test_df: pd.DataFrame, labels_path: str) -> pd.DataFrame:
    labels = load_expected_labels(labels_path)
    if len(labels) != len(test_df):
        raise ValueError(
            f"Expected labels count ({len(labels)}) does not match test rows count ({len(test_df)})"
        )

    labeled_df = test_df.copy()
    labeled_df["label"] = labels
    return labeled_df


def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        df.to_csv(fallback_path, **kwargs)
        print(f"Warning: {path} is locked. Saved to {fallback_path}")
        return fallback_path


def predict_dataframe(
    embedding_model: SentenceTransformer,
    bundle: dict,
    df: pd.DataFrame,
    split_name: str,
) -> tuple[pd.DataFrame, object, object]:
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

    for class_index, class_name in enumerate(label_encoder.classes_):
        result[f"proba_{class_name}"] = probabilities[:, class_index].round(6)

    return result, probabilities, true_ids


def calculate_summary_metrics(result: pd.DataFrame, probabilities, true_ids, class_names) -> dict:
    y_true = result["label"]
    y_pred = result["predicted_label"]
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "log_loss": log_loss(true_ids, probabilities, labels=list(range(len(class_names)))),
    }


def save_classification_report(result: pd.DataFrame, out_path: Path) -> None:
    report = classification_report(
        result["label"],
        result["predicted_label"],
        output_dict=True,
        zero_division=0,
    )
    safe_to_csv(pd.DataFrame(report).transpose(), out_path, encoding="utf-8")


def save_confusion_matrix(result: pd.DataFrame, class_names, split_name: str, out_dir: Path) -> None:
    matrix = confusion_matrix(
        result["label"],
        result["predicted_label"],
        labels=class_names,
    )

    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    safe_to_csv(matrix_df, out_dir / f"confusion_matrix_{split_name}.csv", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", xticks_rotation=25, colorbar=False)
    ax.set_title(f"Confusion Matrix: {split_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_{split_name}.png", dpi=170)
    plt.close(fig)


def save_label_mapping(bundle: dict, out_dir: Path) -> None:
    label_encoder = bundle["label_encoder"]
    mapping = pd.DataFrame(
        {
            "label_id": range(len(label_encoder.classes_)),
            "label": label_encoder.classes_,
        }
    )
    safe_to_csv(mapping, out_dir / "label_mapping.csv", index=False, encoding="utf-8")


def plot_accuracy(summary_df: pd.DataFrame, out_dir: Path, target_accuracy: float | None) -> None:
    values = summary_df["accuracy"].mul(100)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(summary_df["split"], values, color=["#3b82f6", "#16a34a"])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy, %")
    plt.title("Model Accuracy by Dataset Split")

    if target_accuracy is not None:
        target_percent = target_accuracy * 100
        plt.axhline(
            target_percent,
            color="#b42318",
            linestyle="--",
            linewidth=1.4,
            label=f"Target {target_percent:.0f}% (reference)",
        )
        plt.legend(loc="lower right")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_train_test.png", dpi=170)
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
    plt.savefig(out_dir / "predicted_label_distribution.png", dpi=170)
    plt.close()


def print_console_report(
    split_name: str,
    result: pd.DataFrame,
    probabilities,
    true_ids,
    class_names,
    metrics: dict,
) -> None:
    matrix = confusion_matrix(
        result["label"],
        result["predicted_label"],
        labels=class_names,
    )
    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)

    print()
    print("=" * 80)
    print(f"{split_name.upper()} METRICS")
    print("=" * 80)
    print(f"accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"precision_macro:    {metrics['precision_macro']:.4f}")
    print(f"recall_macro:       {metrics['recall_macro']:.4f}")
    print(f"f1_macro:           {metrics['f1_macro']:.4f}")
    print(f"precision_weighted: {metrics['precision_weighted']:.4f}")
    print(f"recall_weighted:    {metrics['recall_weighted']:.4f}")
    print(f"f1_weighted:        {metrics['f1_weighted']:.4f}")
    print(f"log_loss:           {metrics['log_loss']:.4f}")

    print()
    print("Classification report:")
    print(
        classification_report(
            result["label"],
            result["predicted_label"],
            labels=class_names,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix_df.to_string())


def main() -> None:
    default_data_dir = Path.home() / "Desktop" / "llm_guard_dataset"
    default_desktop_dir = Path.home() / "Desktop"
    default_out_dir = Path.home() / "Desktop" / "llm_guard_results"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/guard_model.joblib")
    parser.add_argument("--train-csv", default=str(default_data_dir / "train.csv"))
    parser.add_argument(
        "--test-file",
        default=str(default_desktop_dir / "manual_test.txt"),
        help="TXT with one request per line, or CSV with a 'text' column.",
    )
    parser.add_argument(
        "--test-labels",
        default=str(default_desktop_dir / "manual_label.txt"),
        help="TXT/CSV with expected labels in the same order as test-csv rows.",
    )
    parser.add_argument("--out-dir", default=str(default_out_dir))
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
        help="Draws a clearly labeled target/reference accuracy line on the plot.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model)
    class_names = bundle["label_encoder"].classes_
    embedding_model = SentenceTransformer(bundle["embedding_model_name"])

    train_df = load_dataset(args.train_csv)
    test_raw_df = load_test_texts(args.test_file)
    test_df = attach_expected_labels(test_raw_df, args.test_labels)

    train_results, train_probabilities, train_true_ids = predict_dataframe(
        embedding_model,
        bundle,
        train_df,
        "train",
    )
    test_results, test_probabilities, test_true_ids = predict_dataframe(
        embedding_model,
        bundle,
        test_df,
        "test",
    )
    all_results = pd.concat([train_results, test_results], ignore_index=True)

    summary_rows = []
    for split_name, result, probabilities, true_ids in [
        ("train", train_results, train_probabilities, train_true_ids),
        ("test", test_results, test_probabilities, test_true_ids),
    ]:
        metrics = calculate_summary_metrics(result, probabilities, true_ids, class_names)
        metrics["split"] = split_name
        summary_rows.append(metrics)
        print_console_report(split_name, result, probabilities, true_ids, class_names, metrics)

        save_classification_report(result, out_dir / f"classification_report_{split_name}.csv")
        save_confusion_matrix(result, class_names, split_name, out_dir)

    summary_df = pd.DataFrame(summary_rows)
    ordered_columns = ["split"] + [column for column in summary_df.columns if column != "split"]
    summary_df = summary_df[ordered_columns]

    safe_to_csv(train_results, out_dir / "train_predictions.csv", index=False, encoding="utf-8")
    safe_to_csv(test_results, out_dir / "test_predictions.csv", index=False, encoding="utf-8")
    safe_to_csv(all_results, out_dir / "all_predictions.csv", index=False, encoding="utf-8")
    safe_to_csv(summary_df, out_dir / "metrics_summary.csv", index=False, encoding="utf-8")

    save_label_mapping(bundle, out_dir)
    plot_accuracy(summary_df, out_dir, args.target_accuracy)
    plot_predicted_label_distribution(all_results, out_dir)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"Saved evaluation files to {out_dir}")


if __name__ == "__main__":
    main()
