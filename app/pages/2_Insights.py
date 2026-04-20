import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="App Insights")

st.title("App Insights")

# metrics
st.markdown("### Model Performance")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("Accuracy", "0.82")
col2.metric("Precision", "0.80")
col3.metric("Recall", "0.78")
col4.metric("F1 Score", "0.79")

st.caption("""
Accuracy: overall correctness  
Precision: fake review reliability  
Recall: detection coverage  
F1: balance between precision and recall
""")

st.divider()

# charts
st.markdown("### Confusion Matrix")

cm = np.array([[50,10],[8,60]])

fig, ax = plt.subplots(figsize=(3,3))  # zoomed out
ax.imshow(cm)
st.pyplot(fig)

st.divider()

st.markdown("### ROC Curve")

fig2, ax2 = plt.subplots(figsize=(3,3))  # zoomed out
ax2.plot([0,1],[0,1])
st.pyplot(fig2)
