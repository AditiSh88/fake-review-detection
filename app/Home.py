import streamlit as st
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

st.set_page_config(page_title="AI Review Analyzer", layout="wide")

st.markdown("#AI Review Analyzer")
st.markdown("### Detect Fake Reviews with Explainable AI")

metrics_path = os.path.join(BASE_DIR, "models/metrics.json")

if os.path.exists(metrics_path):
    metrics = json.load(open(metrics_path))
else:
    metrics = {"accuracy":0, "precision":0, "recall":0, "f1":0}

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
col2.metric("Precision", f"{metrics['precision']:.2f}")
col3.metric("Recall", f"{metrics['recall']:.2f}")
col4.metric("F1 Score", f"{metrics['f1']:.2f}")

st.markdown("---")
st.info("Go to Analysis page to test the model.")
