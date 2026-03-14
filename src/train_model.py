import json
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.feature_engineering import build_tfidf_vectorizer
from src.utils import save_object, ensure_dir


def train_model(
    data_path: str = "data/processed/cleaned_dataset.csv",
    model_output_path: str = "models/best_model.pkl",
    metadata_output_path: str = "models/model_metadata.json",
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

    model_configs = {
        "logistic_regression": {
            "pipeline": Pipeline([
                ("tfidf", build_tfidf_vectorizer()),
                ("clf", LogisticRegression(
                    max_iter=3000,
                    solver="liblinear",
                    class_weight="balanced"
                )),
            ]),
            "params": {
                "tfidf__max_features": [3000, 5000, 7000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 3],
                "clf__C": [0.1, 1.0, 10.0],
            },
        },
        "naive_bayes": {
            "pipeline": Pipeline([
                ("tfidf", build_tfidf_vectorizer()),
                ("clf", MultinomialNB()),
            ]),
            "params": {
                "tfidf__max_features": [3000, 5000, 7000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 3],
                "clf__alpha": [0.1, 0.5, 1.0],
            },
        },
        "linear_svm": {
            "pipeline": Pipeline([
                ("tfidf", build_tfidf_vectorizer()),
                ("clf", LinearSVC(class_weight="balanced")),
            ]),
            "params": {
                "tfidf__max_features": [3000, 5000, 7000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 3],
                "clf__C": [0.1, 1.0, 10.0],
            },
        },
    }

    best_model = None
    best_score = -1
    best_name = None
    best_params = None
    all_results = {}

    for model_name, config in model_configs.items():
        print(f"\nRunning GridSearchCV for: {model_name}")

        grid_search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["params"],
            cv=5,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        all_results[model_name] = {
            "best_params": grid_search.best_params_,
            "best_cv_f1": round(float(grid_search.best_score_), 4),
        }

        print(f"Best params for {model_name}: {grid_search.best_params_}")
        print(f"Best CV F1 for {model_name}: {grid_search.best_score_:.4f}")

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_name = model_name
            best_params = grid_search.best_params_

    ensure_dir("models")
    save_object(best_model, model_output_path)

    metadata = {
        "selected_model": best_name,
        "best_cv_f1": round(float(best_score), 4),
        "best_params": best_params,
        "all_model_results": all_results,
    }

    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nOverall Best Model:")
    print(best_name)
    print(f"Best CV F1 Score: {best_score:.4f}")
    print(f"Saved best model to: {model_output_path}")
    print(f"Saved metadata to: {metadata_output_path}")

    return best_model, X_test, y_test, best_name, metadata


if __name__ == "__main__":
    train_model()