from src.data_preprocessing import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.predict import predict_text


def main():
    raw_data_path = "data/raw/pheme_dataset.csv"
    processed_data_path = "data/processed/cleaned_dataset.csv"
    model_path = "models/best_model.pkl"

    print("\nStep 1: Preprocessing dataset...")
    preprocess_data(raw_data_path, processed_data_path)

    print("\nStep 2: Training model...")
    train_model(processed_data_path, model_path)

    print("\nStep 3: Evaluating model...")
    evaluate_model(model_path, processed_data_path)

    print("\nStep 4: Testing a sample prediction...")
    sample_text = "Breaking: official authorities confirmed the emergency update."
    result = predict_text(sample_text, model_path)

    print("\nPrediction Result:")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()