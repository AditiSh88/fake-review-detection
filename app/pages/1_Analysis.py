import streamlit as st
import os
import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from preprocessing import clean_text
from utils.model_loader import load_models
from utils.explain import get_top_words
from utils.report import generate_report

tfidf, logreg = load_models()

st.title("Review Analysis")

text = st.text_area("Enter Review")

if st.button("Analyze"):

    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])

    prob = logreg.predict_proba(vec)[0][1]
    pred = 1 if prob > 0.5 else 0

    if pred == 1:
        st.error("Fake Review")
    else:
        st.success("Genuine Review")

    confidence = prob if pred == 1 else (1 - prob)
    st.progress(float(confidence))

    st.subheader("Important Words")
    words = get_top_words(tfidf, logreg, vec)

    for w, s in words:
        st.write(f"{w} : {s:.3f}")

    st.subheader("WordCloud")
    wc = WordCloud().generate(cleaned)
    fig = plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(fig)

    if st.button("Download Report"):
        file = generate_report(text, pred, confidence)
        with open(file, "rb") as f:
            st.download_button("Download PDF", f, file_name="report.pdf")
