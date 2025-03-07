# Principal Component Analysis (PCA)

## Algorithm Overview

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data from a high-dimensional space into a lower-dimensional space while preserving as much variance as possible. PCA identifies the directions (principal components) in which the data varies the most and projects the data onto these components.

## Implementation Details

This implementation includes:

- **Data Centering** - Subtracting the mean to center the data
- **Covariance Matrix Calculation** - Computing the covariance matrix of the centered data
- **Eigendecomposition** - Finding eigenvalues and eigenvectors of the covariance matrix
- **Sorting Components** - Ordering components by variance (eigenvalue magnitude)
- **Basis Creation** - Selecting top n components to form the projection basis
- **Transformation** - Projecting data onto the new lower-dimensional space

## Mathematical Foundation

PCA works through the following process:
- Computing the covariance matrix: C = (X - μ)ᵀ(X - μ) / (n-1)
- Finding eigenvectors and eigenvalues: Cv = λv
- Sorting eigenvectors by descending eigenvalues
- Creating a projection matrix using the top k eigenvectors
- Transforming the data: X' = (X - μ)P

## Usage Example

```python
# Initialize PCA with desired components
pca = PCA(n_components=2)

# Fit and transform the data
transformed_data = pca.fit_transform(original_data)

# Visualize the dimensionality reduction
plt.scatter(transformed_data[:,0], transformed_data[:,1])
```
## Applications
PCA is useful for:

- Data visualization by reducing data to 2 or 3 dimensions
- Noise reduction by eliminating dimensions with low variance
- Feature engineering to create uncorrelated features
- Speeding up machine learning algorithms by reducing input dimensionality
