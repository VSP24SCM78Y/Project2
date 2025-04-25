import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate_data(n_samples=100, n_features=5, n_informative=3, random_state=42, filename="my_data.csv"):
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state
    )

    # Combine features and target into a DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    generate_data()
