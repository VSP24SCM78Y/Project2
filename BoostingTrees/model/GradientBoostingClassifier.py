import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.initial_pred = 0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        # Convert labels 0/1 to -1/+1
        y = y * 2 - 1

        # Initialize predictions to 0 (log-odds = 0 → prob = 0.5)
        F_m = np.zeros(len(y))

        for m in range(self.n_estimators):
            # Negative gradient for logistic loss
            residuals = y / (1 + np.exp(y * F_m))

            # Fit regression tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Predict and update
            update = tree.predict(X)
            F_m += self.learning_rate * update

            # Store the model
            self.models.append(tree)

    def predict_proba(self, X):
        F_m = np.zeros(X.shape[0])
        for tree in self.models:
            F_m += self.learning_rate * tree.predict(X)
        return self._sigmoid(F_m)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)
