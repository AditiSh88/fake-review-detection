import streamlit as st
import sys, os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Advanced Analysis", layout="wide")

# ✅ Sidebar Styling (consistent across pages)
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a, #4f46e5);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebarNav"] a {
    background: rgba(255,255,255,0.12);
    margin: 6px 10px;
    padding: 10px;
    border-radius: 10px;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(255,255,255,0.25);
}
</style>
""", unsafe_allow_html=True)

# imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

# ---------------- UI ----------------
st.title("Advanced Analysis")

text_input = st.text_area("Enter Review")
model = st.selectbox("Select Model", ["Hybrid", "Logistic Regression", "XGBoost"])

# ---------------- MAIN ACTION ----------------
if st.button("Analyze"):

    vec = tfidf.transform([clean_text(text_input)])

    prob_lr = logreg.predict_proba(vec)[0][1]
    prob_xgb = xgb.predict_proba(vec)[0][1]
    prob_hybrid = (prob_lr + prob_xgb) / 2

    def verdict_box(prob):
        if prob > 0.5:
            return "<div style='background:#fee2e2;padding:12px;border-radius:10px;color:#991b1b'><b>Likely Fake Review</b></div>"
        else:
            return "<div style='background:#dcfce7;padding:12px;border-radius:10px;color:#166534'><b>Likely Genuine Review</b></div>"

    def show_prediction(prob):
        col1, col2 = st.columns(2)
        col1.markdown(f"<div style='background:#dbeafe;padding:10px;border-radius:8px'><b>Prediction Score</b><br>{prob:.3f}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background:#ede9fe;padding:10px;border-radius:8px'><b>Confidence</b><br>{(1-abs(0.5-prob)*2)*100:.2f}%</div>", unsafe_allow_html=True)

    # ---------------- HYBRID ----------------
    if model == "Hybrid":

        st.markdown("## Final Verdict")
        st.markdown(verdict_box(prob_hybrid), unsafe_allow_html=True)

        show_prediction(prob_hybrid)

        # Model Scores
        st.markdown("### Model Scores")
        st.caption("Individual model probabilities contributing to the final hybrid decision.")

        c1, c2 = st.columns(2)
        c1.markdown(f"<div style='background:#fee2e2;padding:10px;border-radius:8px'><b>Logistic Regression</b><br>{prob_lr:.3f}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='background:#e0e7ff;padding:10px;border-radius:8px'><b>XGBoost</b><br>{prob_xgb:.3f}</div>", unsafe_allow_html=True)

        # Explainability
        st.markdown("### Explainability")
        st.caption("Highlighted terms indicate their influence on the model’s decision. Red suggests contribution toward fake classification, while green indicates genuine signals.")

        feature_names = tfidf.get_feature_names_out()
        coefs = logreg.coef_[0]
        vec_arr = vec.toarray()[0]

        weights = {feature_names[i]: coefs[i] for i in range(len(vec_arr)) if vec_arr[i] > 0}

        def highlight(word):
            key = word.lower()
            if key in weights:
                color = "rgba(255,0,0,0.25)" if weights[key] > 0 else "rgba(0,255,0,0.25)"
                return f"<span style='background:{color};padding:3px;border-radius:4px'>{word}</span>"
            return word

        highlighted = " ".join([highlight(w) for w in text_input.split()])
        st.markdown(f"<div style='background:#f8fafc;padding:12px;border-radius:10px'>{highlighted}</div>", unsafe_allow_html=True)

        # Model Comparison Graph
        st.markdown("### Model Comparison")
        st.caption("Comparison of probability outputs from Logistic Regression and XGBoost models.")

        fig, ax = plt.subplots()
        ax.bar(["LogReg", "XGBoost"], [prob_lr, prob_xgb])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

    # ---------------- LOGISTIC REGRESSION ----------------
    elif model == "Logistic Regression":

        st.markdown("## Final Verdict")
        st.markdown(verdict_box(prob_lr), unsafe_allow_html=True)

        show_prediction(prob_lr)

        st.markdown("### About the Model")
        st.write("Logistic Regression is a linear model that evaluates the probability of a review being fake or genuine based on weighted textual features. It is efficient and interpretable, making it suitable for baseline classification tasks.")

    # ---------------- XGBOOST ----------------
    elif model == "XGBoost":

        st.markdown("## Final Verdict")
        st.markdown(verdict_box(prob_xgb), unsafe_allow_html=True)

        show_prediction(prob_xgb)

        st.markdown("### About the Model")
        st.write("XGBoost is an advanced ensemble learning method based on gradient boosting. It builds multiple decision trees sequentially to improve prediction accuracy and capture complex patterns in review data.")

    # ---------------- WORD CLOUD (COMMON FOR ALL) ----------------
    st.markdown("### Word Cloud")
    st.caption("Visual representation of the most prominent words in the review. Larger words indicate higher frequency or importance in the given text.")

    if text_input.strip():
        wc = WordCloud(width=800, height=300, background_color='white').generate(text_input)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
