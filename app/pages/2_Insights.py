st.markdown("## App Insights")

st.markdown("### Model Performance")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.markdown("<div style='background:#e0f2fe;padding:10px;border-radius:8px'>Accuracy: 0.823</div>", unsafe_allow_html=True)
col2.markdown("<div style='background:#fef9c3;padding:10px;border-radius:8px'>Precision: 0.812</div>", unsafe_allow_html=True)
col3.markdown("<div style='background:#dcfce7;padding:10px;border-radius:8px'>Recall: 0.798</div>", unsafe_allow_html=True)
col4.markdown("<div style='background:#ede9fe;padding:10px;border-radius:8px'>F1: 0.805</div>", unsafe_allow_html=True)

st.caption("These metrics evaluate model correctness, reliability, coverage, and balance.")

st.markdown("""
<div style="
background:#e0f2fe;
padding:14px;
border-radius:10px;
color:#0f172a;
line-height:1.5;
">

<b>Confusion Matrix (What it shows)</b><br><br>

This visualization compares the model’s predictions with actual outcomes.<br><br>

It helps you understand:
<ul>
<li>Correctly identified fake and genuine reviews</li>
<li>Misclassified reviews (errors made by the model)</li>
</ul>

In simple terms, it shows how often the model is right or wrong, and where it struggles in real-world detection.

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
background:#ede9fe;
padding:14px;
border-radius:10px;
color:#0f172a;
line-height:1.5;
">

<b>ROC Curve (Model Performance Insight)</b><br><br>

This graph shows how well the model separates fake and genuine reviews across different decision thresholds.<br><br>

It explains:
<ul>
<li>How effectively the model detects fake reviews</li>
<li>How well it avoids wrongly flagging genuine reviews</li>
</ul>

A curve closer to the top-left corner means the model is more accurate and reliable in distinguishing between the two classes.

</div>
""", unsafe_allow_html=True)
