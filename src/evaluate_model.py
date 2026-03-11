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
)

from src.train_model import train_model
from src.utils import ensure_dir


def save_top_features(model, output_path: str, top_n: int = 20) -> None:
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    feature_names = np.array(tfidf.get_feature_names_out())
    coefficients = clf.coef_[0]

    top_real_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_rumor_idx = np.argsort(coefficients)[:top_n]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Top features indicating NON-RUMOR / TRUE class\n")
        f.write("=" * 50 + "\n")
        for idx in top_real_idx:
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


def evaluate_model(
    model_path: str = "models/best_model.pkl",
    data_path: str = "data/processed/cleaned_dataset.csv",
):
    ensure_dir("results")

    model, X_test, y_test = train_model(data_path, model_path)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(y_test, y_pred, target_names=["Non-Rumor", "Rumor"])
    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "results/confusion_matrix.png")

    save_top_features(model, "results/top_features.txt")

    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification Report:\n")
    print(report)


if __name__ == "__main__":
    evaluate_model()