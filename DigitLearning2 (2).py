#https://www.kaggle.com/c/digit-recognizer
#https://www.youtube.com/watch?v=aZsZrkIgan0&t=502s

#scikit learn is a popular ML library

import numpy as np
from sklearn.cross_validation import train_test_split

#used to plot image
import matplotlib.pyplot as pt

#to load training data set
import pandas as pd

from sklearn.tree import DecisionTreeClassifier




#data includes label (what the digit is) and its various pixels
#convert into 2d numpy array (READ ABOUT NUMPY AND PANDAS)
data = pd.read_csv("DigitData/train.csv").as_matrix()


clf = DecisionTreeClassifier()



### TRAINING DATASET

X = data[0:, 1:]
y = data[0:, 0]

#.5 = separate into halves
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=.5)


#train our classifier
clf.fit(x_train,y_train)





predicitions = clf.predict(x_test)


from sklearn.metrics import accuracy_score
print("Decision Tree: ")
print(accuracy_score(y_test, predicitions))



