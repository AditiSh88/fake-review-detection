from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf():
    return TfidfVectorizer(max_features=5000)

def transform_tfidf(vectorizer, X):
    return vectorizer.transform(X)
