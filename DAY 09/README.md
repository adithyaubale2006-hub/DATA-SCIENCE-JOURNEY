# Advanced Feature Engineering & Dimensionality Reduction

This project demonstrates end-to-end advanced data preprocessing and feature selection techniques used in high-stakes industries like Quantitative Finance and E-commerce. The goal is to maximize predictive performance by capturing non-linear relationships and reducing data noise.

## Highlights

- **Advanced Feature Engineering**: Implemented Log Transformations, Polynomial Features, and Interaction Terms to capture complex data patterns.
- **Feature Selection & Importance**: Utilized Random Forest Regressors to rank features and identify high-impact variables.
- **Dimensionality Reduction**: Applied Principal Component Analysis (PCA) to reduce high-dimensional feature sets while retaining >95% of data variance.
- **Data Pipelines**: Built robust workflows including missing value imputation, categorical binning, and standard scaling.

## Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Environment**: Google Colab / Jupyter Notebook

## Key Results

- **PCA Analysis**: Successfully reduced 6 complex financial features into 2 principal components explaining **97.73%** of total variance.
- **Model Insights**: Identified 'Annual Income' and 'Debt-to-Income Ratio' as the primary drivers for loan amount predictions via Feature Importance ranking.

## Contents

1. **Data Cleaning**: Handling missing numerical and categorical data.
2. **Transformations**: Applying `log1p` to normalize skewed distributions.
3. **Feature Creation**: Generating interaction effects (e.g., Credit Score × Employment Years).
4. **Dimensionality Reduction**: Visualization of data clusters using PCA.

---
