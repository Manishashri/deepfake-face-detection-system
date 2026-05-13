import numpy as np

class MultiHeadLDA:
    def __init__(self, heads=4):
        self.heads = heads
        self.models = []

    def transform(self, X):
        outputs = []
        for lda, s, e in self.models:
            outputs.append(lda.transform(X[:, s:e]))
        return np.concatenate(outputs, axis=1)