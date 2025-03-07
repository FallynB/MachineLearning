#!/usr/bin/env python
# coding: utf-8

# **Homework 1**

# We begin with the usual import, and a new one:

# In[19]:


import numpy as np
from sklearn.datasets import load_iris


# Now load the iris dataset.

# In[22]:


iris=load_iris()
X=iris.data 
y=iris.target


# The columns of the numpy array `X` (our "feature matrix") give the Sepal Length, Sepal Width, Petal Length and Petal Width of 150 different observed iris flowers. `y` is our "target", an array of 150 integers indicating the specific species of iris, where 0=Setosa, 1=Versicolor, and 2=Virginica.
# 
# Here are the first few rows of `X`:

# In[25]:


X[:5,:]


# For this assignment, we'll only work with the Petal Length and Petal Width of each flower, so we can redefine `X` to be just the last two columns:

# In[28]:


X=X[:,2:]
X.shape


# Define a function `sq_distances` with inputs `X` (a numpy array with two columns), `length` and `width` (the Petal Length and Petal Width of an unknown flower). The function should return an array of squared distances from the unknown point to each point in `X`. Use vectorized Numpy operations, NOT A FOR LOOP. 

# In[39]:


def sq_distances(X,length,width):
    location = np.array([length, width])
    calculation = np.sum((X - location) ** 2, axis =1)
    return calculation


# Define a function `SpeciesOfKNeighbors` that gives the species label (a number 0, 1, or 2) of the k nearest neighbors from the point with given Petal Length and Petal Width to the points in `X`. (The list of species labels for each point in `X` is contained in the array `y`.) *Hint: The numpy function `argsort()` is useful for this problem.*

# In[77]:


def SpeciesOfNeighbors(X,y,length,width,k):
    distances = sq_distances(X,length,width)
    indicies = np.argsort(distances)[:k]
    return y[indicies]
    


# Create a function `majority` that takes an array of labels, and returns the label that appears the most often. *Hint: The numpy functions `bincount()` and `argmax()` can be useful here.*

# In[54]:


def majority(labels):
    return np.argmax(np.bincount(labels))


# Combine your previous functions to create a function `KNN` which takes a feature matrix `X` of known Petal Lengths and Petal Widths, a target array `y` containing their species labels, a hyperparameter `k`, and the `length` and `width` of the petal of an unknown flower. Your function should return the most common species index among the k nearest neighbors of the unknown flower. 

# In[70]:


def KNN(X,y,length,width,k):
    neighbor = SpeciesOfNeighbors(X,y,length,width,k)
    return majority(neighbor)
    


# Test your code by playing with a few values for length, width, and k. For example, try:

# In[79]:


KNN(X,y,1,1,7)


# Moving forward, we'll write our ML models as classes that conform to the standards of the sklearn package. Let's do this now. Modify your functions above to create appropriate methods for the following class:

# In[87]:


class KNeighborsClassifier():
    def __init__(self,k):
        self.n_neighbors=k

    def fit(self,X,y):
        self.X=X
        self.y=y

    def sq_distances(self,length,width):
        location = np.array([length, width])
        calculation = np.sum((self.X - location) ** 2, axis =1)
        return calculation 
        #YOUR CODE HERE

    def SpeciesOfNeighbors(self,length,width):
        distances = self.sq_distances(length, width)
        indicies = np.argsort(distances) [:self.n_neighbors]
        return self.y[indicies]

    def majority(self,labels):
        return np.argmax(np.bincount(labels))

    def predict(self,length, width):
        neighbor = self.SpeciesOfNeighbors(length,width)
        return self.majority(neighbor)


# If done correctly, the following code should produce the same answer as before:

# In[90]:


knn=KNeighborsClassifier(7)
knn.fit(X,y)
knn.predict(1,1)


# In[ ]:





# In[ ]:




