import shap

def explain_model(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.plots.bar(shap_values)
