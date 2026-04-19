import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from feature_engineering import get_tfidf
from preprocessing import load_data

df = load_data("data/reviews.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42
)

tfidf = get_tfidf()
X_train_tfidf = tfidf.fit_transform(X_train)

logreg = LogisticRegression()
logreg.fit(X_train_tfidf, y_train)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_tfidf, y_train)

pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))
pickle.dump(logreg, open("models/logreg.pkl", "wb"))
pickle.dump(xgb, open("models/xgb.pkl", "wb"))
