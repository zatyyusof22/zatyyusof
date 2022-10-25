# -*- coding: utf-8 -*-
"""unsupervise.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ofuzd4iKLc0hWw49z4mYFooFJcLFa4KB
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

iris = sns.load_dataset('iris') 
X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

xtrain, xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state = 0)

from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 1], X[:,1], s=50);

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 1], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 1], c='black', s=200, alpha=0.5);