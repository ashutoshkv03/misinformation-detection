from src.utils import load_object, clean_text


def predict_text(text: str, model_path: str = "models/best_model.pkl") -> dict:
    model = load_object(model_path)

    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]

    label = "RUMOR / MISINFORMATION" if prediction == 1 else "NON-RUMOR / TRUE"
    confidence = float(max(probabilities))

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "predicted_label": label,
        "confidence": round(confidence, 4),
    }


if __name__ == "__main__":
    sample_text = "Reports say a miracle cure was discovered overnight with no proof."
    result = predict_text(sample_text)
    print(result)