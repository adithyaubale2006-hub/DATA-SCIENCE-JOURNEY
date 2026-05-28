# Day 10: Hyperparameter Tuning, Cross-Validation, and ML Pipelines

Welcome to Day 10 of the Machine Learning Curriculum. This repository covers production-level model optimization, workflow automation, and robust validation strategies using `scikit-learn`.

---

##  Key Topics Covered

### 1. Hyperparameters Deep Dive
Understanding the external levers that control model learning before training begins. 
* **Underfitting vs. Overfitting:** Comparing simple configurations (`max_depth=2`) against complex ones (`max_depth=20`) to visualize how model capacity impacts performance.

### 2. Automated Hyperparameter Tuning
Moving away from manual trial-and-error by implementing automated search algorithms:
* **GridSearchCV:** Exhaustive search across a specified parameter grid.
* **RandomizedSearchCV:** A faster, scalable alternative that samples fixed random combinations from a broader distribution space.

### 3. K-Fold Cross Validation
The gold standard for evaluating model stability. Rotating validation splits across $K$ folds ensures the model's accuracy is dependable and not just a byproduct of a "lucky" train-test split.

### 4. Machine Learning Pipelines
Structuring code like a Senior ML Engineer. Using `sklearn.pipeline.Pipeline` to encapsulate preprocessing transformations (like `StandardScaler`) and estimators into a single atomic workflow to strictly prevent **data leakage**.

### 5. Advanced Evaluation Metrics
Going beyond basic accuracy to analyze model performance using:
* Precision, Recall, and F1-Score (for imbalanced classification tasks).
* Confusion Matrix analysis using `seaborn` heatmaps.

---

##  Code Architecture

The notebook emphasizes the industry-standard design pattern where a data preprocessing pipeline is nested directly inside a hyperparameter search grid:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 1. Bundle scaling and modeling together
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# 2. Target specific pipeline steps using the double underscore (__) prefix
param_grid = {
    'rf__n_estimators': [100, 300],
    'rf__max_depth': [5, 10, None],
    'rf__min_samples_split': [2, 5]
}

# 3. Cross-validate the entire workflow
grid_search = GridSearchCV(pipeline, param_grid, cv=3)
grid_search.fit(X_train, y_train)
