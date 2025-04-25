# Project 2  
**Course:** CS584 - Machine Learning  
**Instructor:** Steve Avsec  

## Team Member  
Rohith Kukkadapu – A20554359  
Vinay Yerram – A20554778  
Rama Muni Reddy Bandi – A20554387  
Kumar Sri Pavan Veeramallu – A20539662  

---

## Gradient Boosting Classifier from Scratch

---

### 🔹 What does the model you have implemented do and when should it be used?

This project implements a **Gradient Boosting Classifier** from scratch using decision stumps (shallow trees) as weak learners. Gradient boosting is an ensemble technique where models are built sequentially to correct the errors of their predecessors, optimizing a specific loss function — in this case, **logistic loss** for binary classification.

It should be used when:
- You are solving binary classification problems with complex, non-linear decision boundaries  
- Your dataset is tabular and benefits from tree-based methods  
- You require probabilistic outputs (e.g., for ranking or thresholding)  
- You aim to minimize log loss or improve metrics like AUC using boosting techniques

---

### 🔹 How did you test your model to determine if it is working reasonably correctly?

- Generated synthetic classification data using `sklearn.datasets.make_classification`
- Split data into train and test sets using `train_test_split`
- Evaluated model performance using `accuracy_score` and `classification_report`
- Verified `.predict()` returns class labels and `.predict_proba()` returns probabilities between 0 and 1
- Built a test script (`test_BoostingClassifier.py`) to automate training and report ~90%+ test accuracy
- Ran the full pipeline in a Jupyter notebook to visually inspect and debug intermediate steps

---

### 🔹 What parameters have you exposed to users of your implementation in order to tune performance?

- `n_estimators` – Number of boosting rounds (i.e., weak learners)  
- `learning_rate` – Step size multiplier that controls each learner’s contribution  
- `max_depth` – Depth of each decision tree (default is 1, i.e., decision stump)

Basic usage example:
```python
from BoostingTrees.model.GradientBoostingClassifier import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

### 🔹 Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

- **Multi-class classification**: The model currently handles only binary classification. With more time, this could be extended using one-vs-rest schemes or softmax-based losses.  
- **Imbalanced datasets**: Boosting may overfit to the majority class. Future work could involve class weighting or sampling techniques.  
- **Noisy data**: Boosting can overfit noise. Improvements such as early stopping, feature sampling, or regularization could address this.  
- **High-cardinality features**: Although trees don't need feature scaling, skewed or high-cardinality features may lead to biased splits. Preprocessing or binning could help.


---

###  How to Run the Project

####  Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

####  Run the Models

##### Run Gradient Boosting Classifier from Scratch

```bash
cd model
python GradientBoostingClassifier.py
```

```python
from BoostingTrees.model.GradientBoostingClassifier import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample classification data
X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

#### 📊 View Visualizations (Optional)

```bash
cd demo
jupyter notebook BoostingTrees_Demo.ipynb
```

You’ll see:
- Accuracy and classification report
- Predicted probability distributions
- Training and prediction examples using your model

---

#### 🧪 Run Tests

```bash
cd tests
python test_BoostingClassifier.py
```

This runs:
- End-to-end model training on generated data
- Accuracy evaluation and sanity checks on predictions
