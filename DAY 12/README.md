#  Day 12: Deep Learning & Neural Networks Basics

Welcome to Day 12 of the machine learning journey! This repository contains a comprehensive introduction to Deep Learning, foundational Neural Network theory, and hands-on implementations using **TensorFlow** and **Keras**.

---

##  Table of Contents

1. [Overview](https://www.google.com/search?q=%23-overview)
2. [Theoretical Concepts Covered](https://www.google.com/search?q=%23-theoretical-concepts-covered)
3. [Key Implementations](https://www.google.com/search?q=%23-key-implementations)
* [1. Single-Neuron Linear Regression](https://www.google.com/search?q=%231-single-neuron-linear-regression)
* [2. Multi-Layer ANN: Customer Purchase Prediction](https://www.google.com/search?q=%232-multi-layer-ann-customer-purchase-prediction)


4. [Tech Stack & Dependencies](https://www.google.com/search?q=%23-tech-stack--dependencies)
5. [How to Run](https://www.google.com/search?q=%23-how-to-run)
6. [Key Takeaways](https://www.google.com/search?q=%23-key-takeaways)

---

##  Overview

This project serves as a step-by-step pipeline moving from the fundamental mathematical unit of deep learning—the individual neuron—to a fully functional, multi-layer **Artificial Neural Network (ANN)** engineered for binary classification.

---

##  Theoretical Concepts Covered

* **Machine Learning vs. Deep Learning:** Understanding how deep learning automates feature extraction from unstructured data using dense hidden layers.
* **The Anatomy of a Neuron:** Breaking down the foundational unit mapping an input vector to an output:

$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)$$


* **Weights & Biases:** The learnable parameters adjusted during optimization to reduce error.
* **Activation Functions:** Introducing non-linearity via **ReLU**, **Sigmoid**, and **Tanh** to enable networks to learn highly complex patterns.
* **The Training Loop:** Understanding the relationship between **Forward Propagation** (making predictions), **Loss Functions** (measuring error via MSE or Cross-Entropy), **Backpropagation** (calculating gradients via the calculus chain rule), and **Gradient Descent** (updating parameters dynamically based on the learning rate $\alpha$).

---

##  Key Implementations

### 1. Single-Neuron Linear Regression

A foundational workflow demonstrating a single neuron learning the simple linear formula $y = 2x - 1$. This script utilizes the modern Keras `Input` standard to safely build the model graph without lifecycle warnings.

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense

# 1. Prepare data mapping the formula: y = 2x - 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# 2. Define model architecture using explicit Input layer
model = Sequential([
    Input(shape=(1,)),  # Accept 1 numerical feature
    Dense(units=1)      # Single neuron tracking a linear transformation
])

# 3. Compile using Stochastic Gradient Descent (SGD) and Mean Squared Error (MSE)
model.compile(optimizer='sgd', loss='mean_squared_error')

# 4. Train the model 
print("Training the single-neuron model...")
model.fit(xs, ys, epochs=500, verbose=0)

# 5. Infer on unseen data (Target value for 10.0 is 19.0)
prediction = model.predict(np.array([10.0]))
print(f"Prediction for input 10.0: {prediction[0][0]:.4f}")

```

### 2. Multi-Layer ANN: Customer Purchase Prediction

A production-grade architecture that stacks multiple `Dense` hidden layers utilizing `ReLU` activations, terminating in a single-neuron output layer with a `Sigmoid` activation for binary classification (predicting whether a customer will buy a product based on **Age** and **Income**).

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense

# 1. Synthesize customer records
data = {
    "Age": [22, 25, 30, 35, 40, 28, 32, 45],
    "Income": [25000, 30000, 40000, 50000, 70000, 35000, 45000, 80000],
    "Purchased": [0, 0, 0, 1, 1, 0, 1, 1]
}
df = pd.DataFrame(data)
X = df[["Age", "Income"]]
y = df["Purchased"]

# 2. Architect a Deep Neural Network 
model = Sequential([
    Input(shape=(2,)),                  # Two input dimensions (Age & Income)
    Dense(16, activation="relu"),       # Hidden Layer 1: 16 units with non-linear ReLU
    Dense(8, activation="relu"),        # Hidden Layer 2: 8 units with non-linear ReLU
    Dense(1, activation="sigmoid")      # Output Layer: 1 unit mapping a 0-1 probability
])

# 3. Compile with Adam optimizer and Binary Cross-Entropy
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 4. Inspect Network Structure
model.summary()

# 5. Fit the model
print("\nTraining the Multi-Layer Network...")
model.fit(X, y, epochs=10, verbose=1)

# 6. Evaluate inference on custom profiles
new_customer = np.array([[33.0, 48000.0]])
prob = model.predict(new_customer)
result = "Likely to Purchase" if prob[0][0] > 0.5 else "Unlikely to Purchase"
print(f"\nPrediction for Age 33, Income $48k: {prob[0][0]:.4f} -> {result}")

```

---

##  Tech Stack & Dependencies

* **Language:** Python 3
* **Deep Learning Framework:** TensorFlow 2.x (via the integrated Keras API)
* **Data Processing:** NumPy, Pandas

---




> 
>
