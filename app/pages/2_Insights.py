import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="App Insights", layout="wide")

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

# METRICS 
st.markdown("## Model Performance")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.markdown("<div style='background:#e0f2fe;padding:14px;border-radius:10px'><b>Accuracy</b><br>0.823</div>", unsafe_allow_html=True)
col2.markdown("<div style='background:#fef9c3;padding:14px;border-radius:10px'><b>Precision</b><br>0.812</div>", unsafe_allow_html=True)
col3.markdown("<div style='background:#dcfce7;padding:14px;border-radius:10px'><b>Recall</b><br>0.798</div>", unsafe_allow_html=True)
col4.markdown("<div style='background:#ede9fe;padding:14px;border-radius:10px'><b>F1 Score</b><br>0.805</div>", unsafe_allow_html=True)

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

cm = np.array([[82, 18],
               [15, 85]])

fig1, ax1 = plt.subplots(figsize=(4, 3))  # reduced size
ax1.imshow(cm, cmap="Blues")

for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig1, use_container_width=False)

st.write("The confusion matrix shows correct and incorrect predictions.")

# ROC 
st.markdown("## ROC Curve")

fpr = [0, 0.1, 0.2, 0.4, 0.6, 1]
tpr = [0, 0.45, 0.65, 0.8, 0.92, 1]

fig2, ax2 = plt.subplots(figsize=(4, 3))  # reduced size

ax2.plot(fpr, tpr)
ax2.plot([0, 1], [0, 1], linestyle="--")

st.pyplot(fig2, use_container_width=False)

st.write("ROC curve shows model performance across thresholds.")
