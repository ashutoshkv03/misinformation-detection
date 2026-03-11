import pandas as pd
from src.utils import clean_text, ensure_dir


def preprocess_data(
    input_path: str = "data/raw/pheme_dataset.csv",
    output_path: str = "data/processed/cleaned_dataset.csv",
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required_columns = ["text", "is_rumor"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    keep_columns = ["text", "is_rumor"]
    optional_columns = ["user.handle", "topic"]

    for col in optional_columns:
        if col in df.columns:
            keep_columns.append(col)

    df = df[keep_columns].copy()
    df = df.dropna(subset=["text", "is_rumor"])

    df["text"] = df["text"].astype(str)
    df["label"] = df["is_rumor"].astype(int)
    df["cleaned_text"] = df["text"].apply(clean_text)

    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)

    ensure_dir("data/processed")
    df.to_csv(output_path, index=False)

    print(f"Preprocessed dataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")
    print(df[["text", "cleaned_text", "label"]].head())

    return df


if __name__ == "__main__":
    preprocess_data()