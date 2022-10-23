

import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

xtrain, xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state = 0)

clf = SVC(kernel='rbf', C=1).fit(xtrain, ytrain)
st.write('Iris dataset')
st.write('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format(clf.score(xtrain, ytrain)))
st.write('Accuracy of RBF SVC classifier on test set: {:.2f}'
     .format(clf.score(xtest, ytest)))

from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

svm = SVC(random_state=42, kernel='linear')

# Fit the data to the SVM classifier
svm = svm.fit(xtrain, ytrain)

# Evaluate by means of a confusion matrix
matrix = plot_confusion_matrix(svm, xtest, ytest, cmap=plt.cm.Blues, normalize='true')
plt.title('Confusion matrix for linear SVM')
plt.show(matrix)
plt.show()

from sklearn.metrics import classification_report
from sklearn.svm import SVC
model = SVC()                       
model.fit(xtrain, ytrain)                  
y_model = model.predict(xtest)

cr = classification_report(ytest, y_model)
st.write(cr)