import streamlit as st

st.title("App Insights")

# ---------------- METRICS ----------------
st.markdown("### Model Performance")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("Accuracy", "0.823")
col2.metric("Precision", "0.812")
col3.metric("Recall", "0.798")
col4.metric("F1 Score", "0.805")

st.caption("These metrics show how accurate and balanced the model is overall.")

# ---------------- CONFUSION MATRIX ----------------
st.markdown("### Confusion Matrix")

st.write("The confusion matrix shows how many predictions were correct or incorrect by comparing predicted and actual labels.")
st.write("It helps understand where the model makes mistakes in classifying fake and genuine reviews.")

# (plot here remains unchanged)

# ---------------- ROC ----------------
st.markdown("### ROC Curve")

st.write("The ROC curve shows how well the model separates fake and genuine reviews at different thresholds.")
st.write("A curve closer to the top-left indicates better classification performance.")

# (plot here remains unchanged)
