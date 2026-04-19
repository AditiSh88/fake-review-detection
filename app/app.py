import streamlit as st
import pickle
from preprocessing import clean_text

tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
logreg = pickle.load(open("models/logreg.pkl", "rb"))

st.title("Fake Review Detection System")

text = st.text_area("Enter Review")

if st.button("Predict"):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = logreg.predict(vec)[0]

    if pred == 1:
        st.error("Fake Review Detected")
    else:
        st.success("Genuine Review")
