import numpy as np
from src.utils import load_object, clean_text


def get_prediction_scores(model, cleaned_text: str):
    clf = model.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        probabilities = model.predict_proba([cleaned_text])[0]
        return {
            "non_rumor_score": float(probabilities[0]),
            "rumor_score": float(probabilities[1]),
            "confidence": float(max(probabilities)),
            "score_type": "probability",
        }

    if hasattr(clf, "decision_function"):
        raw_score = float(model.decision_function([cleaned_text])[0])
        pseudo_prob = 1 / (1 + np.exp(-raw_score))

        return {
            "non_rumor_score": float(1 - pseudo_prob),
            "rumor_score": float(pseudo_prob),
            "confidence": float(max(1 - pseudo_prob, pseudo_prob)),
            "score_type": "sigmoid_decision_score",
        }

    return {
        "non_rumor_score": None,
        "rumor_score": None,
        "confidence": None,
        "score_type": "unavailable",
    }


def predict_text(text: str, model_path: str = "models/best_model.pkl") -> dict:
    model = load_object(model_path)

    cleaned = clean_text(text)
    prediction = int(model.predict([cleaned])[0])

    scores = get_prediction_scores(model, cleaned)

    label = "RUMOR / MISINFORMATION" if prediction == 1 else "NON-RUMOR / TRUE"

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "predicted_numeric_label": prediction,
        "predicted_label": label,
        "non_rumor_score": round(scores["non_rumor_score"], 4) if scores["non_rumor_score"] is not None else None,
        "rumor_score": round(scores["rumor_score"], 4) if scores["rumor_score"] is not None else None,
        "confidence": round(scores["confidence"], 4) if scores["confidence"] is not None else None,
        "score_type": scores["score_type"],
    }


if __name__ == "__main__":
    sample_text = "Reports say a miracle cure was discovered overnight with no proof."
    result = predict_text(sample_text)
    print(result)