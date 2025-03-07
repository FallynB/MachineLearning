#!/usr/bin/env python
# coding: utf-8

# **Homework 3**
# 
# Load the iris dataset:

# In[1]:


import numpy as np


# Create a PCA class that is instantiated by specifying the number of desired components.

# In[35]:


class PCA():
  def __init__(self,n_components):
      self.n_components=n_components
      self.meann = None
      

  def fit(self,X):
      self.meann = np.mean(X, axis = 0)
      centeredX = X - self.meann #Center the feature matrix about its mean (column-wise)
      CVmatrix = np.cov(centeredX, rowvar=False) #Create the CV matrix from centeredX


      
      #Compute the eigenvals and eigenvecs of CVmatrix using built in numpy libraries, pref
      eigvals,eigvecs = np.linalg.eig(CVmatrix)
      
      indecies_sort = np.argsort(eigvals)[:: -1]
      eigvals = eigvals[indecies_sort]
      eigvecs = eigvecs[:, indecies_sort]
      
      signs = np.sign(eigvecs[np.argmax(np.abs(eigvecs), axis = 0), range(eigvecs.shape[1])])
      eigvecs = eigvecs * signs[np.newaxis, :]
      
      #Create a basis from the eigenvecs with the largest corresponding eigenvals:
      self.basis = eigvecs[:, :self.n_components]
      
  def transform(self,X):
      #Project X onto the basis created by the fit method
      centerX = X - self.meann
      return np.dot(centerX, self.basis)

  def fit_transform(self,X):
    self.fit(X)
    return self.transform(X) #Combines the fit method and the transform method for convenience


# The following code block loads the  `iris` dataset and applies a `PCA` object with 2 components.

# In[38]:


from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
y=iris.target

pca=PCA(n_components=2)
projectedX=pca.fit_transform(X)


# Run this code block to visualize your projection! Note that the color of each point comes from the species, allowing you to see to what extent those points form distinct clusters.

# In[41]:


import matplotlib.pyplot as plt

plt.scatter(projectedX[:,0],projectedX[:,1],c=y)

