import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add path to 'Boosting trees' folder for import
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# Import from the actual file path (even with space)
from model.GradientBoostingClassifier import GradientBoostingClassifier

def main():
    df = pd.read_csv(os.path.join(project_root, "my_data.csv"))
    X = df.drop("target", axis=1).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

