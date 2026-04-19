import streamlit as st
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

metrics = json.load(open(os.path.join(BASE_DIR, "models/metrics.json")))

st.title("📊 Model Insights")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", metrics["accuracy"])
col2.metric("Precision", metrics["precision"])
col3.metric("Recall", metrics["recall"])
col4.metric("F1 Score", metrics["f1"])

st.markdown("---")

# Example visuals
y_true = [0,1,0,1,0,1,0,1]
y_pred = [0,1,0,1,0,0,0,1]

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
fig = plt.figure()
ConfusionMatrixDisplay(cm).plot()
st.pyplot(fig)

st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

fig2 = plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1])
plt.legend()
st.pyplot(fig2)
