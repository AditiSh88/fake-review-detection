st.markdown("## Advanced Analysis")

model = st.selectbox("Select Model", ["Hybrid", "Logistic Regression", "XGBoost"])

prob_lr, prob_xgb = 0.7, 0.6
prob_hybrid = (prob_lr + prob_xgb) / 2

def verdict(p):
    return "Fake" if p > 0.5 else "Genuine"

# hybrid
if model == "Hybrid":

    st.markdown("### Final Hybrid Result")

    st.markdown(f"""
    <div style='background:#dbeafe;padding:10px;border-radius:8px'>
    Verdict: {verdict(prob_hybrid)}<br>
    Prediction: {prob_hybrid:.3f}
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    col1.markdown(f"<div style='background:#fee2e2;padding:10px;border-radius:8px'>LogReg: {prob_lr:.3f}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='background:#e0e7ff;padding:10px;border-radius:8px'>XGBoost: {prob_xgb:.3f}</div>", unsafe_allow_html=True)

# model specific
elif model == "Logistic Regression":
    st.markdown("### Logistic Regression Analysis Only")
    st.write("Prediction:", prob_lr)
    st.write("Verdict:", verdict(prob_lr))

elif model == "XGBoost":
    st.markdown("### XGBoost Analysis Only")
    st.write("Prediction:", prob_xgb)
    st.write("Verdict:", verdict(prob_xgb))

# shap
st.markdown("### SHAP Explainability")
st.caption("Feature contribution visualization for selected model")

# (your SHAP code stays here)

# word cloud
st.markdown("### Word Cloud")
