import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="App Insights")

st.title("App Insights")

# dummy data (replace if needed)
cm = np.array([[50, 10],[8, 60]])

st.markdown("### Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", "0.82")
col2.metric("Precision", "0.80")
col3.metric("Recall", "0.78")

st.divider()

# smaller charts
col4, col5 = st.columns(2)

with col4:
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(cm)
    st.pyplot(fig)

with col5:
    fig2, ax2 = plt.subplots(figsize=(3,3))
    ax2.plot([0,1],[0,1])
    st.pyplot(fig2)
