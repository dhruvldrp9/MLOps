import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings(action='ignore')

iris=pd.read_csv("iris.csv")

X = iris['Sepal.Length'].values.reshape(-1,1)
Y = iris['Sepal.Width'].values.reshape(-1,1)

corr_mat = iris.corr()

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


train, test = train_test_split(iris, test_size = 0.25)


train_X = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
train_y = train.Species

test_X = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
test_y = test.Species

#Using LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)


# prediction = model.predict(test_X)
# print('Accuracy:',metrics.accuracy_score(prediction,test_y))

# from sklearn.metrics import confusion_matrix,classification_report
# confusion_mat = confusion_matrix(test_y,prediction)
# print("Confusion matrix: \n",confusion_mat)
# print(classification_report(test_y,prediction))