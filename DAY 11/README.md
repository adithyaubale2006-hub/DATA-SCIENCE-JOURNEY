# Unsupervised Learning: Customer Segmentation & Anomaly Detection

A comprehensive project exploring unsupervised machine learning techniques to derive insights from unlabeled customer data.

##  Overview
This repository contains the implementation of core unsupervised learning algorithms to segment customer behavior and identify anomalies in transactional data.

## 🛠 Tech Stack
* **Python**
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-learn** (K-Means, PCA, Isolation Forest)
* **Matplotlib** (Visualization)

##  Key Techniques Implemented

| Technique | Application |
| :--- | :--- |
| **K-Means Clustering** | Grouping customers based on Income vs. Spending behavior. |
| **Elbow Method** | Mathematically determining the optimal number of clusters ($K$). |
| **PCA** | Dimensionality reduction for visualization and feature optimization. |
| **Isolation Forest** | Anomaly/Fraud detection to identify unusual data points. |

## 📈 Key Findings
* **Segmentation:** Successfully categorized customers into distinct groups (e.g., High-income low-spenders vs. Budget-conscious high-spenders).
* **Anomaly Detection:** Flagged outliers that deviate from standard behavioral patterns, crucial for fraud prevention.
* **Optimization:** Used the Elbow Method to validate the selection of $K=3$ for the sample dataset.

##  Project Structure
- `DAY 11.ipynb`: Full implementation, including data preparation, scaling, modeling, and visualization.
- `data/`: Sample datasets used for clustering and anomaly training.

##  Contributing
Feel free to open an issue or submit a pull request if you have ideas to optimize the clustering parameters or implement new anomaly detection models!

## 📝 License
Distributed under the MIT License.
