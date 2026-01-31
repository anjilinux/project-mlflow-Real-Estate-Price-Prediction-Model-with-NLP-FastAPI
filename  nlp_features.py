from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=500,
    stop_words="english"
)
X_text = vectorizer.fit_transform(df['description'])