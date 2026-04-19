import os
import re
import pickle
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier

import nltk
from nltk.corpus import stopwords

# download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# load data
def load_data(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.lower().str.strip()

    if 'review' not in df.columns:
        raise ValueError("Column 'review' not found")

    if 'label' not in df.columns:
        raise ValueError("Column 'label' not found")

    if df['label'].dtype == 'object':
        df['label'] = df['label'].astype(str).str.lower().str.strip()
        df['label'] = df['label'].replace({
            'fake': 1,
            'real': 0,
            'cg': 1,
            'or': 0
        })

    df = df[['review', 'label']].dropna()
    df['label'] = df['label'].astype(int)
    df['review'] = df['review'].apply(clean_text)

    return df

# MAIN
def main():
    print("Loading dataset...")
    df = load_data("data/reviews.csv")

    X = df['review']
    y = df['label']

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_tfidf, y_train)

    # XGBoost
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1
    )
    xgb.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred_lr = logreg.predict(X_test_tfidf)
    y_pred_xgb = xgb.predict(X_test_tfidf)

    print("\n--- Logistic Regression ---")
    print(classification_report(y_test, y_pred_lr))

    print("\n--- XGBoost ---")
    print(classification_report(y_test, y_pred_xgb))

    # save models
    os.makedirs("models", exist_ok=True)

    pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))
    pickle.dump(logreg, open("models/logreg.pkl", "wb"))
    pickle.dump(xgb, open("models/xgb.pkl", "wb"))

    print("\nModels saved successfully")

    # save metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_xgb)),
        "precision": float(precision_score(y_test, y_pred_xgb)),
        "recall": float(recall_score(y_test, y_pred_xgb)),
        "f1": float(f1_score(y_test, y_pred_xgb))
    }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Metrics saved successfully")

# run
if __name__ == "__main__":
    main()
