from sklearn.metrics import classification_report, roc_auc_score

def evaluate(model, X, y):
    preds = model.predict(X)
    print(classification_report(y, preds))
