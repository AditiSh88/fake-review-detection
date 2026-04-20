import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="App Insights", layout="wide")

st.title("App Insights")

# ---------------- MODEL PERFORMANCE METRICS ----------------
st.markdown("## Model Performance")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.markdown(
    "<div style='background:#e0f2fe;padding:14px;border-radius:10px'><b>Accuracy</b><br>0.823</div>",
    unsafe_allow_html=True,
)

col2.markdown(
    "<div style='background:#fef9c3;padding:14px;border-radius:10px'><b>Precision</b><br>0.812</div>",
    unsafe_allow_html=True,
)

col3.markdown(
    "<div style='background:#dcfce7;padding:14px;border-radius:10px'><b>Recall</b><br>0.798</div>",
    unsafe_allow_html=True,
)

col4.markdown(
    "<div style='background:#ede9fe;padding:14px;border-radius:10px'><b>F1 Score</b><br>0.805</div>",
    unsafe_allow_html=True,
)

st.caption("""
Accuracy measures overall correctness of predictions.
Precision shows how many predicted fake reviews were actually fake.
Recall measures how many actual fake reviews were correctly detected.
F1 Score balances precision and recall for fair evaluation.
""")

# ---------------- CONFUSION MATRIX ----------------
st.markdown("## Confusion Matrix")

cm = np.array([[82, 18],
               [15, 85]])

fig1, ax1 = plt.subplots()

ax1.imshow(cm, cmap="Blues")

for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm[i, j], ha="center", va="center")

ax1.set_xlabel("Predicted Label")
ax1.set_ylabel("Actual Label")
ax1.set_title("Confusion Matrix")

st.pyplot(fig1)

st.write("""
The confusion matrix compares actual vs predicted results.
It shows how many reviews were correctly classified and where the model made mistakes.
This helps evaluate real-world reliability of the system.
""")

# ---------------- ROC CURVE ----------------
st.markdown("## ROC Curve")

fpr = [0, 0.1, 0.2, 0.4, 0.6, 1]
tpr = [0, 0.45, 0.65, 0.8, 0.92, 1]

fig2, ax2 = plt.subplots()

ax2.plot(fpr, tpr, label="Model ROC Curve")
ax2.plot([0, 1], [0, 1], linestyle="--")

ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend()

st.pyplot(fig2)

st.write("""
The ROC curve shows how well the model separates fake and genuine reviews across different thresholds.
A curve closer to the top-left corner indicates better performance and higher classification ability.
""")
