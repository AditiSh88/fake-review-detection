import streamlit as st

st.set_page_config(page_title="Advanced Analysis")

st.title("Advanced Analysis")

text = st.text_area("Enter review")

if st.button("Analyze"):

    st.markdown("### Additional Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Word Count", len(text.split()))
    col2.metric("!", text.count("!"))
    col3.metric("Uppercase", sum(1 for w in text.split() if w.isupper()))

    st.markdown("### Explanation")

    st.markdown("""
    <div style="background:#f9fafb;padding:15px;border-radius:10px;color:#111;">
    Explainability shown here
    </div>
    """, unsafe_allow_html=True)
