import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def load_models():
    tfidf = pickle.load(open(os.path.join(BASE_DIR, "models/tfidf.pkl"), "rb"))
    logreg = pickle.load(open(os.path.join(BASE_DIR, "models/logreg.pkl"), "rb"))
    xgb = pickle.load(open(os.path.join(BASE_DIR, "models/xgb.pkl"), "rb"))
    return tfidf, logreg, xgb
