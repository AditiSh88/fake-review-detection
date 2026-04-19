import streamlit as st
import os
import sys
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import shap

# Fix path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from preprocessing import clean_text
from utils.model_loader import load_models
from utils.explain import get_top_words
from utils.report import generate_report

tfidf, logreg, xgb = load_models()

st.set_page_config(layout="wide")

# header
st.title("🚀 AI Review Analyzer")

st.markdown("---")

# input
text = st.text_area("Enter Review", height=120)

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "XGBoost"]
)

analyze = st.button("Analyze Review")

# analysis
if analyze and text.strip() != "":
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])

    model = logreg if model_choice == "Logistic Regression" else xgb

    prob = model.predict_proba(vec)[0][1]
    pred = 1 if prob > 0.5 else 0
    confidence = prob if pred == 1 else (1 - prob)

    
    # ui cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📌 Prediction")
        if pred == 1:
            st.error("⚠️ Fake Review")
        else:
            st.success("✅ Genuine Review")

    with col2:
        st.markdown("### 📊 Confidence")
        st.progress(float(confidence))
        st.write(f"{confidence:.2f}")

    st.markdown("---")

    # explainability
    st.subheader("🔍 Important Words")

    words = get_top_words(tfidf, logreg, vec)

    for w, s in words:
        st.write(f"**{w}** → {s:.3f}")

    st.markdown("---")

    # shap explainability
    st.subheader("🧠 SHAP Explainability")

    try:
        explainer = shap.Explainer(model, vec)
        shap_values = explainer(vec)

        fig = plt.figure()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP not supported for this model input format.")

    st.markdown("---")

    # wordcloud
    st.subheader("☁️ WordCloud")

    wc = WordCloud().generate(cleaned)
    fig_wc = plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(fig_wc)

    st.markdown("---")

    # model comparison
    st.subheader("⚖️ Model Comparison")

    prob_lr = logreg.predict_proba(vec)[0][1]
    prob_xgb = xgb.predict_proba(vec)[0][1]

    col3, col4 = st.columns(2)

    col3.metric("Logistic Regression", f"{prob_lr:.2f}")
    col4.metric("XGBoost", f"{prob_xgb:.2f}")

    st.markdown("---")

    # report
    if st.button("📄 Generate Report"):
        file = generate_report(text, pred, confidence)
        with open(file, "rb") as f:
            st.download_button("Download PDF", f, file_name="report.pdf")
