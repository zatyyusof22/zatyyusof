import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.header("My first Streamlit App")
st.write(pd.DataFrame({
    'Intplan': ['yes', 'yes', 'yes', 'no'],
    'Churn Status': [0, 0, 0, 1]
}))
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



from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
iris = sns.load_dataset('iris')

X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xtrain, ytrain)


fig = plt.figure(figsize=(10, 4))
clf.fit(Xtrain, ytrain)

tree.plot_tree(clf.fit(Xtrain, ytrain) )

clf.score(Xtest, ytest)
