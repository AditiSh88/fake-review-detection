st.markdown("## App Insights")

st.markdown("### Model Performance")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.markdown("<div style='background:#e0f2fe;padding:10px;border-radius:8px'>Accuracy: 0.823</div>", unsafe_allow_html=True)
col2.markdown("<div style='background:#fef9c3;padding:10px;border-radius:8px'>Precision: 0.812</div>", unsafe_allow_html=True)
col3.markdown("<div style='background:#dcfce7;padding:10px;border-radius:8px'>Recall: 0.798</div>", unsafe_allow_html=True)
col4.markdown("<div style='background:#ede9fe;padding:10px;border-radius:8px'>F1: 0.805</div>", unsafe_allow_html=True)

st.caption("These metrics evaluate model correctness, reliability, coverage, and balance.")

st.markdown("### Confusion Matrix")
st.caption("""
The confusion matrix shows how well the model is performing by comparing its predictions with the actual results.

It tells us:
- how many reviews were correctly identified as fake or genuine
- and how many were incorrectly classified

In simple terms, it helps us understand where the model is making mistakes and how reliable it is in real-world decisions.
""")

st.markdown("### ROC Curve")
st.caption("""
The ROC curve shows how well the model can distinguish between fake and genuine reviews at different decision thresholds.

It helps us understand:
- how sensitive the model is in detecting fake reviews
- and how well it avoids incorrectly flagging genuine reviews

A curve closer to the top-left corner means the model is performing better and making more accurate decisions overall.
""")
