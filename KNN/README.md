# K-Nearest Neighbors (KNN) Algorithm

## Algorithm Overview

K-Nearest Neighbors (KNN) is a simple, versatile, non-parametric supervised learning algorithm used for classification. It operates on the principle that similar data points exist in close proximity to each other, making predictions based on the majority class of the k closest data points.

## Implementation Details

This implementation of KNN works with the Iris dataset and includes:

1. Distance calculation using vectorized operations for efficiency
2. Neighbor identification based on Euclidean distance
3. Majority voting mechanism
4. A scikit-learn compatible class interface

## Mathematical Foundation

For a new data point with features (length, width), the algorithm:
1. Calculates the squared Euclidean distance to all training points: d² = Σ(x_i - p_i)²
2. Identifies the k points with the smallest distances
3. Assigns the most common class label among these k neighbors

## Usage Example

```python
# Initialize the classifier
knn = KNeighborsClassifier(k=7)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
prediction = knn.predict(length=1.0, width=1.0)
