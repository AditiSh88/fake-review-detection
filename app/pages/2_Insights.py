import streamlit as st
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

metrics = json.load(open(os.path.join(BASE_DIR, "models/metrics.json")))

st.title("Model Insights")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", metrics["accuracy"])
col2.metric("Precision", metrics["precision"])
col3.metric("Recall", metrics["recall"])
col4.metric("F1 Score", metrics["f1"])

st.markdown("---")

# sample data (replace later if needed)
y_true = [0,1,0,1,0,1,0,1]
y_pred = [0,1,0,1,0,0,0,1]

st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(2, 2))
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(ax=ax)
st.pyplot(fig)

st.markdown("---")

st.subheader("ROC Curve")

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots(figsize=(3, 2)
ax2.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
ax2.plot([0,1],[0,1])
ax2.legend()
st.pyplot(fig2)
