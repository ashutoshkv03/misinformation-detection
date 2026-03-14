import os
import re
import pickle


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_object(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_object(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)