# DAY 6 — Machine Learning Basics (Quick Notes)

## What is Machine Learning?

Machine Learning = teaching computers to learn patterns from data and make predictions.

Example:

* house price prediction
* spam detection
* forex forecasting

---

# Types of ML

## 1. Supervised Learning

Has:

* input (X)
* output (y)

### Types

* Regression → predicts numbers
* Classification → predicts categories

---

## 2. Unsupervised Learning

No output labels.

Used for:

* clustering
* segmentation

---

# Features vs Target

| Term         | Meaning           |
| ------------ | ----------------- |
| Features (X) | input columns     |
| Target (y)   | output to predict |

Example:

* Experience = X
* Salary = y

---

# Train-Test Split

Used to:

* train model
* test on unseen data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

# Linear Regression

Used for predicting numbers.

Formula:

y = mx + b

Example:

* salary prediction
* sales forecasting

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

---

# Predictions

```python
predictions = model.predict(X_test)
```

---

# Logistic Regression

Used for classification.

Examples:

* spam/not spam
* fraud/not fraud
* buy/sell

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
```

---

# Overfitting vs Underfitting

| Type         | Meaning                   |
| ------------ | ------------------------- |
| Underfitting | model too simple          |
| Overfitting  | model memorizes data      |
| Good Model   | performs well on new data |

---

# Regression Metrics

## MAE

$$\text{MAE} = \frac{1}{n}\sum |y - \hat{y}|$$

---

## MSE

$$\text{MSE} = \frac{1}{n}\sum (y - \hat{y})^2$$

---

## RMSE

$$\text{RMSE} = \sqrt{\text{MSE}}$$

---

# Classification Metric

## Accuracy

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

---

# Basic ML Workflow

```text
Collect Data
   ↓
Clean Data
   ↓
Train Model
   ↓
Predict
   ↓
Evaluate
```

---

# Minimal ML Example

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "Experience": [1,2,3,4,5],
    "Salary": [30000,40000,50000,60000,70000]
}

df = pd.DataFrame(data)

X = df[["Experience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)
```

---

# End Goal of Day 6

Understand ML basics
Know regression & classification
Train simple ML models
Make predictions
Understand overfitting
Use evaluation metrics
