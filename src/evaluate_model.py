import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import train_test_split

from src.utils import ensure_dir, load_object


def save_top_features(model, output_path: str, top_n: int = 20) -> None:
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    if not hasattr(clf, "coef_"):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Top feature extraction is not supported for this classifier.\n")
        return

    feature_names = np.array(tfidf.get_feature_names_out())
    coefficients = clf.coef_[0]

    top_non_rumor_idx = np.argsort(coefficients)[:top_n]
    top_rumor_idx = np.argsort(coefficients)[-top_n:][::-1]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Top features indicating NON-RUMOR / TRUE class\n")
        f.write("=" * 50 + "\n")
        for idx in top_non_rumor_idx:
            f.write(f"{feature_names[idx]}: {coefficients[idx]:.4f}\n")

        f.write("\nTop features indicating RUMOR / MISINFORMATION class\n")
        f.write("=" * 50 + "\n")
        for idx in top_rumor_idx:
            f.write(f"{feature_names[idx]}: {coefficients[idx]:.4f}\n")


def plot_confusion_matrix(cm, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Non-Rumor", "Rumor"],
        yticklabels=["Non-Rumor", "Rumor"],
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(y_test, y_scores, output_path: str) -> float:
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return float(pr_auc)


def get_model_scores(model, X_test):
    clf = model.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(clf, "decision_function"):
        raw_scores = model.decision_function(X_test)
        raw_scores = np.asarray(raw_scores, dtype=float)

        min_val = raw_scores.min()
        max_val = raw_scores.max()

        if max_val - min_val == 0:
            return np.zeros_like(raw_scores)

        return (raw_scores - min_val) / (max_val - min_val)

    return None


def evaluate_model(
    model_path: str = "models/best_model.pkl",
    data_path: str = "data/processed/cleaned_dataset.csv",
):
    ensure_dir("results")

    model = load_object(model_path)

    df = pd.read_csv(data_path)
    if "cleaned_text" not in df.columns or "label" not in df.columns:
        raise ValueError("Processed dataset must contain 'cleaned_text' and 'label' columns.")

    X = df["cleaned_text"]
    y = df["label"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    y_pred = model.predict(X_test)
    y_scores = get_model_scores(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }

    if y_scores is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_scores))
        metrics["pr_auc"] = plot_precision_recall_curve(
            y_test,
            y_scores,
            "results/precision_recall_curve.png"
        )

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Non-Rumor", "Rumor"]
    )
    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "results/confusion_matrix.png")

    with open("results/confusion_matrix_values.txt", "w", encoding="utf-8") as f:
        f.write(str(cm))

    save_top_features(model, "results/top_features.txt")

    results_df = pd.DataFrame({
        "text": X_test.values,
        "true_label": y_test.values,
        "predicted_label": y_pred,
    })
    misclassified = results_df[results_df["true_label"] != results_df["predicted_label"]]
    misclassified.to_csv("results/misclassified_examples.csv", index=False)

    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification Report:\n")
    print(report)
    print("\nSaved outputs to results/")


if __name__ == "__main__":
    evaluate_model()