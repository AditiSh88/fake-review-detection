import streamlit as st
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

metrics = json.load(open(os.path.join(BASE_DIR, "models/metrics.json")))

st.title("Model Insights")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", metrics["accuracy"])
col2.metric("Precision", metrics["precision"])
col3.metric("Recall", metrics["recall"])
col4.metric("F1 Score", metrics["f1"])
