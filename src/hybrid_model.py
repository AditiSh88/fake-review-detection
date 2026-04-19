import numpy as np

class HybridModel:
    def __init__(self, model1, model2, w1=0.5, w2=0.5):
        self.model1 = model1
        self.model2 = model2
        self.w1 = w1
        self.w2 = w2

    def predict(self, X1, X2):
        p1 = self.model1.predict_proba(X1)[:,1]
        p2 = self.model2.predict_proba(X2)[:,1]
        final = self.w1 * p1 + self.w2 * p2
        return (final > 0.5).astype(int)

    def predict_proba(self, X1, X2):
        p1 = self.model1.predict_proba(X1)[:,1]
        p2 = self.model2.predict_proba(X2)[:,1]
        return self.w1 * p1 + self.w2 * p2
