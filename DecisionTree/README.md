# Decision Tree Classifier

## Algorithm Overview

A Decision Tree is a supervised learning algorithm that creates a tree-like model of decisions based on feature values. It builds a flowchart-like structure where each internal node represents a test on a feature, each branch represents an outcome of that test, and each leaf node represents a class label.

## Implementation Details

This implementation builds a decision tree classifier from scratch with the following components:

- **Gini Impurity** - A measure of how often a randomly chosen element would be incorrectly labeled if labeled randomly according to the distribution of labels in the subset
- **Split Evaluation** - Calculation of weighted Gini impurity for potential splits
- **Best Split Finding** - Examination of all possible feature/threshold combinations to find optimal splits
- **Tree Construction** - Recursive building of the tree structure with nodes and leaves
- **Prediction Logic** - Traversing the tree to make predictions on new data points

## Node Structure

Each node in the decision tree contains:
- Whether it's a leaf node
- If leaf: the majority class label
- If internal node: the feature and threshold for splitting
- If internal node: references to left and right child nodes

## Gini Impurity

The Gini impurity is calculated as:
- G = 1 - Σ(pi²) where pi is the probability of an item being classified to class i

When evaluating a split, the weighted average of the Gini impurity of the resulting subsets is used:
- G_split = (n_left/n_total) * G_left + (n_right/n_total) * G_right

## Usage Example

```python
# Initialize the classifier with maximum depth
tree = DecisionTreeClassifier(max_depth=5)

# Train the model
tree.fit(X_train, y_train)

# Make predictions
prediction = tree.predict(x_new)

# Evaluate accuracy
accuracy = (y_test == predictions).mean()
```


## Advantages

- Intuitive and easy to explain
- Requires little data preprocessing
- Can handle both numerical and categorical data
- Implicitly performs feature selection
- Non-parametric, making few assumptions about the data distribution
