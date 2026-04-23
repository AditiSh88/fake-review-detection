import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import pandas as pd

from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="App Insights", layout="wide")

# SIDEBAR STYLE
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a, #4f46e5);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.title("App Insights")

# LOAD METRICS 
with open("models/metrics.json", "r") as f:
    metrics = json.load(f)

# LOAD MODELS
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
logreg = pickle.load(open("models/logreg.pkl", "rb"))
xgb = pickle.load(open("models/xgb.pkl", "rb"))

# LOAD DATA
df = pd.read_csv("data/reviews.csv")
df.columns = df.columns.str.lower().str.strip()

X = df["review"]
y = df["label"]

# Transform
X_tfidf = tfidf.transform(X)

# HYBRID PREDICTIONS
prob_lr = logreg.predict_proba(X_tfidf)[:, 1]
prob_xgb = xgb.predict_proba(X_tfidf)[:, 1]

prob_hybrid = (prob_lr + prob_xgb) / 2
y_pred = (prob_hybrid > 0.5).astype(int)

# METRICS 
st.markdown("## Model Performance")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.markdown(f"<div style='background:#e0f2fe;padding:14px;border-radius:10px'><b>Accuracy</b><br>{metrics['hybrid']['accuracy']:.3f}</div>", unsafe_allow_html=True)
col2.markdown(f"<div style='background:#fef9c3;padding:14px;border-radius:10px'><b>Precision</b><br>{metrics['hybrid']['precision']:.3f}</div>", unsafe_allow_html=True)
col3.markdown(f"<div style='background:#dcfce7;padding:14px;border-radius:10px'><b>Recall</b><br>{metrics['hybrid']['recall']:.3f}</div>", unsafe_allow_html=True)
col4.markdown(f"<div style='background:#ede9fe;padding:14px;border-radius:10px'><b>F1 Score</b><br>{metrics['hybrid']['f1']:.3f}</div>", unsafe_allow_html=True)

st.markdown("""
<div style='color:#1f2937;font-size:14px'>
Accuracy measures overall correctness of predictions.<br><br>
Precision shows how many predicted fake reviews were actually fake.<br><br>
Recall measures how many actual fake reviews were correctly detected.<br><br>
F1 Score balances precision and recall for fair evaluation.
</div>
""", unsafe_allow_html=True)

# CONFUSION MATRIX
st.markdown("## Confusion Matrix")

cm = confusion_matrix(y, y_pred)

fig1, ax1 = plt.subplots(figsize=(10, 9))
ax1.imshow(cm, cmap="Blues")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax1.text(j, i, cm[i, j], ha="center", va="center")

ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")

st.pyplot(fig1)

st.write("The confusion matrix shows how many predictions were correct and incorrect across classes.")

# ROC CURVE
st.markdown("## ROC Curve")

fpr, tpr, _ = roc_curve(y, prob_hybrid)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots(figsize=(10, 9))
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax2.plot([0, 1], [0, 1], linestyle="--")

ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()

st.pyplot(fig2)

st.write("ROC curve shows how well the model separates fake and genuine reviews across thresholds.")
