import streamlit as st
import os, sys
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import shap

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from preprocessing import clean_text
from utils.model_loader import load_models
from utils.explain import get_top_words
from utils.report import generate_report

tfidf, logreg, xgb = load_models()

st.set_page_config(layout="wide")

st.markdown("""
<style>
.fade-in { animation: fadeIn 0.8s ease-in; }
@keyframes fadeIn {
from {opacity:0; transform: translateY(10px);}
to {opacity:1; transform: translateY(0);}
}
.card {
padding:20px;
border-radius:12px;
background:#111827;
box-shadow:0 4px 20px rgba(0,0,0,0.3);
transition:0.3s;
}
.card:hover {
transform:translateY(-5px);
}
</style>
""", unsafe_allow_html=True)
st.title("🔍 Review Analysis")

text = st.text_area("Enter Review", height=120)

model_choice = st.selectbox("Model", ["Logistic Regression", "XGBoost"])

analyze = st.button("Analyze Review")

if analyze and text.strip() != "":
    
    with st.spinner("Analyzing... 🤖"):
        cleaned = clean_text(text)
        vec = tfidf.transform([cleaned])

        model = logreg if model_choice == "Logistic Regression" else xgb

        prob = model.predict_proba(vec)[0][1]
        pred = 1 if prob > 0.5 else 0
        confidence = prob if pred == 1 else (1 - prob)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.subheader("📌 Prediction")
        if pred == 1:
            st.error("Fake Review")
        else:
            st.success("Genuine Review")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.subheader("📊 Confidence")
        st.progress(float(confidence))
        st.write(f"{confidence:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🔍 Important Words")
    words = get_top_words(tfidf, logreg, vec)

    for w, s in words:
        st.write(f"{w} → {s:.3f}")

    st.markdown("---")

    st.subheader("🧠 SHAP Explainability")

    try:
        explainer = shap.Explainer(model, vec)
        shap_values = explainer(vec)

        fig = plt.figure()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

    except:
        st.warning("SHAP not supported for this model.")

    st.markdown("---")

    st.subheader("☁️ WordCloud")

    wc = WordCloud().generate(cleaned)
    fig = plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("⚖️ Model Comparison")

    prob_lr = logreg.predict_proba(vec)[0][1]
    prob_xgb = xgb.predict_proba(vec)[0][1]

    col3, col4 = st.columns(2)
    col3.metric("LogReg", f"{prob_lr:.2f}")
    col4.metric("XGBoost", f"{prob_xgb:.2f}")

    st.markdown("---")

    if st.button("Generate Report"):
        file = generate_report(text, pred, confidence)
        with open(file, "rb") as f:
            st.download_button("Download PDF", f, file_name="report.pdf")
