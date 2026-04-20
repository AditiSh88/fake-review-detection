import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="App Insights")

st.title("App Insights")

st.markdown("### Model Performance")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("Accuracy", "0.82")
col2.metric("Precision", "0.80")
col3.metric("Recall", "0.78")
col4.metric("F1 Score", "0.79")

st.caption("""
Accuracy shows overall correctness. Precision indicates reliability of fake predictions.
Recall measures how many fake reviews were detected. F1 score balances both precision and recall.
""")

st.divider()

# CONFUSION MATRIX
st.markdown("### Confusion Matrix")
st.caption("Shows correct vs incorrect classifications made by the model.")

cm = np.array([[50, 10],[8, 60]])

fig, ax = plt.subplots(figsize=(1.5,1.5))
ax.imshow(cm)
st.pyplot(fig)

st.divider()

# ROC
st.markdown("### ROC Curve")
st.caption("Represents the trade-off between true positive rate and false positive rate.")

fig2, ax2 = plt.subplots(figsize=(1.5,1.5))
ax2.plot([0,1],[0,1])
st.pyplot(fig2)
