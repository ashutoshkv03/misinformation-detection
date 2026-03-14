from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words="english",
):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        stop_words=stop_words,
    )


def create_tfidf_features(
    train_texts,
    test_texts,
    max_features: int = 5000,
    ngram_range=(1, 2),
):
    vectorizer = build_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer