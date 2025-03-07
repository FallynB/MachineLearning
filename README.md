# Machine Learning Algorithms

This repository contains implementations of various machine learning algorithms created as part of a Machine Learning course. Each algorithm is implemented from scratch to demonstrate understanding of the underlying concepts.
The implementations in this repository are based on the foundations and principles taught in CSCI158 PZ-01, and I would like to acknowledge and thank my professor for providing the theoretical framework and guidance that made these implementations possible.

## Algorithms Included

**K-Nearest Neighbors (KNN)**: A non-parametric method that classifies objects based on the majority class of their k nearest neighbors in the feature space. KNN makes no assumptions about the underlying data distribution, making it adaptable to complex datasets.

**Principal Component Analysis (PCA)**: A dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving maximum variance. PCA identifies the axes (principal components) that capture the most information in the data, making it useful for visualization and addressing the curse of dimensionality.

**Decision Tree**: A tree-based model that makes decisions by recursively splitting the data based on feature values to create homogeneous subsets. Decision trees are interpretable models that naturally handle non-linear relationships and feature interactions.

**Random Forest**: An ensemble method that combines multiple decision trees to improve prediction accuracy and reduce overfitting. Random forests introduce randomness through bootstrap sampling and feature subset selection, creating diverse trees whose collective wisdom outperforms individual models.

**Gradient Boosting Regressor**: A sequential ensemble technique that builds trees iteratively, with each tree correcting the errors made by previous trees. Gradient boosting focuses on minimizing residuals through gradient descent, resulting in powerful models that excel at capturing complex patterns.

**Linear Regression**: A fundamental algorithm that models the relationship between variables by finding the linear equation that minimizes the sum of squared errors. Linear regression serves as a building block for more complex models and provides interpretable coefficients that quantify feature importance.

**Polynomial Regression**: An extension of linear regression that captures non-linear relationships by including polynomial terms of input features. Polynomial regression enables modeling curved relationships while maintaining the interpretability advantages of linear models.

**Gradient Descent**: An iterative optimization algorithm that finds the minimum of a function by taking steps proportional to the negative of the gradient. Gradient descent underpins many machine learning algorithms by providing a mechanism to efficiently minimize loss functions.

**Stochastic Gradient Descent**: A more efficient version of gradient descent that updates parameters using randomly selected subsets of data. SGD significantly reduces computational requirements while often converging faster than traditional gradient descent, making it essential for training models on large datasets.

## Machine Learning and Data Science

Machine learning provides the algorithms and statistical models that enable computer systems to improve performance on specific tasks through experience without explicit programming, while data science encompasses the broader field of extracting knowledge and insights from data using various techniques from statistics, computer science, and domain expertise. Together, these disciplines form the backbone of modern analytics, powering everything from recommendation systems to medical diagnostics through their ability to uncover patterns in data and translate them into actionable insights.

* **Stochastic Gradient Descent**: A more efficient version of gradient descent.

## Repository Structure

Each algorithm is organized in its own directory

## Datasets Used

The implementations use several classic machine learning datasets:
* Iris dataset
* Wine dataset
* California Housing dataset
* Cars dataset

## Getting Started

To use these algorithms, you'll need:
* Python 3.6+
* NumPy
* Matplotlib
* scikit-learn (for dataset loading and comparison)
  
