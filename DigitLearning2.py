#https://www.kaggle.com/c/digit-recognizer
#https://www.youtube.com/watch?v=aZsZrkIgan0&t=502s

#scikit learn is a popular ML library

import numpy as np

#used to plot image
import matplotlib.pyplot as pt

#to load training data set
import pandas as pd

from sklearn.tree import DecisionTreeClassifier


#data includes label (what the digit is) and its various pixels
#convert into 2d numpy array (READ ABOUT NUMPY AND PANDAS)
data = pd.read_csv("DigitData/train.csv").as_matrix()


#print(data)

clf = DecisionTreeClassifier()



### TRAINING DATASET
#CHANGE WITH SPECIFIC FUNCTION TO SPLIT DATA
#rows 0-21000, everything except label
xtrain = data[0:21000, 1:]
#train label is data's 0th column, so only contains label
train_label = data[0:21000,0]


#train our classifier
clf.fit(xtrain,train_label)




### TESTING DATASET
xtest = data[21000:,1:]
actual_label = data[21000:,0]





from PIL import Image
filename = "digit8.jpg"
im = Image.open(filename,'r')
pix_val = list(im.getdata())
pix_val_flat = [x for sets in pix_val for x in sets]







### LETS PREDICT!
#####visualize a random element in test data 
d = pix_val_flat
d.shape=(28,28)
pt.imshow(255-d,cmap='gray')

#####predict what it is
print(clf.predict( [d] ))

#####visualize d
pt.show()


#makes list of predicitions for the xtest data set
p = clf.predict(xtest)

#REPLACE WITH VALIDITY TEST
count = 0
for i in range(0,21000):
    if p[i] == actual_label[i]:
        count +=1

print("Accuracy=", (count/21000)*100) 