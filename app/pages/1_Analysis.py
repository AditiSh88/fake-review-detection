import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Advanced Analysis")

st.title("Advanced Analysis")

text = st.text_area("Enter Review")

model = st.selectbox("Select Model", ["Hybrid", "Logistic Regression", "XGBoost"])

if st.button("Analyze"):

    prob_lr = 0.7
    prob_xgb = 0.6
    prob_hybrid = (prob_lr + prob_xgb) / 2

    def verdict(p):
        return "Fake" if p > 0.5 else "Genuine"

    # metrics
    st.markdown("### Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Hybrid", prob_hybrid)
    col2.metric("LogReg", prob_lr)
    col3.metric("XGBoost", prob_xgb)

    st.divider()

    # verdict
    st.markdown("### Model Results")

    st.write("Logistic Regression:", verdict(prob_lr))
    st.write("XGBoost:", verdict(prob_xgb))
    st.write("Hybrid:", verdict(prob_hybrid))

    st.divider()

    # wordcloud
    st.markdown("### Word Importance")

    wc = WordCloud(background_color="white").generate(text)

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)
