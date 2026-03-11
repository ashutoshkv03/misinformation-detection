import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.feature_engineering import build_tfidf_vectorizer
from src.utils import save_object, ensure_dir


def train_model(
    data_path: str = "data/processed/cleaned_dataset.csv",
    model_output_path: str = "models/best_model.pkl",
):
    df = pd.read_csv(data_path)

    if "cleaned_text" not in df.columns or "label" not in df.columns:
        raise ValueError("Processed dataset must contain 'cleaned_text' and 'label' columns.")

    X = df["cleaned_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear")),
    ])

    param_grid = {
        "tfidf__max_features": [3000, 5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    ensure_dir("models")
    save_object(best_model, model_output_path)

    print("\nBest Parameters:")
    print(grid_search.best_params_)
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
    print(f"Saved model to: {model_output_path}")

    return best_model, X_test, y_test


if __name__ == "__main__":
    train_model()