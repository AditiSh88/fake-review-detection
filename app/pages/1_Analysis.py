import streamlit as st
import os, sys
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import shap

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

st.set_page_config(layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 2rem; }

.card {
    padding:18px;
    border-radius:12px;
    background:#111827;
    box-shadow:0 4px 20px rgba(0,0,0,0.3);
    margin-bottom:15px;
}

.badge {
    display:inline-block;
    padding:6px 10px;
    border-radius:8px;
    margin-right:8px;
    font-size:14px;
}

.word:hover {
    outline:1px solid white;
    cursor:pointer;
}
</style>
""", unsafe_allow_html=True)

st.title("Review Analysis")

text = st.text_area("Enter Review", height=120)
model_choice = st.selectbox("Model", ["Logistic Regression", "XGBoost"])

if st.button("Analyze") and text.strip() != "":

    with st.spinner("Analyzing... "):
        cleaned = clean_text(text)
        vec = tfidf.transform([cleaned])

        model = logreg if model_choice == "Logistic Regression" else xgb

        prob = model.predict_proba(vec)[0][1]
        pred = 1 if prob > 0.5 else 0
        confidence = prob if pred == 1 else (1 - prob)
  
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction")
        st.success("Genuine Review") if pred == 0 else st.error("Fake Review")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Confidence")
        st.progress(float(confidence))
        st.write(f"{confidence:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    st.subheader("Explainability")

    feature_names = tfidf.get_feature_names_out()
    coefs = logreg.coef_[0]

    vec_array = vec.toarray()[0]
    indices = vec_array.nonzero()[0]

    word_weights = {feature_names[i]: coefs[i] for i in indices}
    max_weight = max(abs(v) for v in word_weights.values()) if word_weights else 1

    def highlight(word):
        w = word.lower()
        if w in word_weights:
            score = word_weights[w]
            norm = score / max_weight

            color = f"rgba(255,0,0,{abs(norm)})" if norm > 0 else f"rgba(0,255,0,{abs(norm)})"

            return f"<span class='word' title='Score: {score:.3f}' style='background:{color};padding:3px 6px;border-radius:5px'>{word}</span>"
        return word

    highlighted = " ".join([highlight(w) for w in text.split()])

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Original**")
        st.markdown(f"<div class='card'>{text}</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("**Highlighted**")
        st.markdown(f"<div class='card'>{highlighted}</div>", unsafe_allow_html=True)

    st.markdown("""
    <span class="badge" style="background:rgba(0,255,0,0.4)">🟢 Genuine</span>
    <span class="badge" style="background:rgba(255,0,0,0.4)">🔴 Fake</span>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- SHAP ----------
    st.subheader("SHAP Explainability")

    try:
        feature_names = tfidf.get_feature_names_out()
        explainer = shap.Explainer(model)
        shap_values = explainer(vec)
        shap_values.feature_names = feature_names

        fig = plt.figure(figsize=(6,3))
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

    except:
        st.warning("SHAP not available.")

    st.markdown("---")

    st.subheader("WordCloud")

    wc = WordCloud().generate(cleaned)
    fig = plt.figure(figsize=(5,3))
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(fig)

    st.markdown("---")
-
    st.subheader("Model Comparison")

    prob_lr = logreg.predict_proba(vec)[0][1]
    prob_xgb = xgb.predict_proba(vec)[0][1]

    col3, col4 = st.columns(2)
    col3.metric("LogReg", f"{prob_lr:.2f}")
    col4.metric("XGBoost", f"{prob_xgb:.2f}")
