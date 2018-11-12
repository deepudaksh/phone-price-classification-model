#-----------------------------------------------------------------------------------------------------------------------------------------------#
#Program: Supervised Classification Model for Phone price classification.
#Developed by: Pradeep L
#Date: 07 October 2018
#Tags: SVM, Kernel, Poly fucnction, Supervise model, Classification
#-----------------------------------------------------------------------------------------------------------------------------------------------#

#Importing Libraries.
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC   
from sklearn.metrics import classification_report, confusion_matrix  

#Reading Datasets for Training and Test.
trainingData = pd.read_csv("train.csv")  
testData = pd.read_csv("test.csv")  

#Seperating Target outcome along with other attributes.
X = trainingData.drop('price_range', axis=1)
y = trainingData['price_range'] 

#Splitting data into Training and Test data. (Training: 80%, Test: 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  


#Building classification model. (Poly function)
# svclassifier = SVC(kernel='linear')  
svclassifier = SVC(kernel='poly', degree=4) 
svclassifier.fit(X_train, y_train)  

#Predicting for split data and Checking the accuracy.
# y_pred = svclassifier.predict(X_test)
# print(confusion_matrix(y_test,y_pred))  
# print(classification_report(y_test,y_pred))

#Preparing Test Data for Prediction.
testData = testData.drop('id', axis=1)
predictValues = svclassifier.predict(testData)

#Creating List of Index
c= 0
idList = []
while c<1000:
    c+=1
    idList.append(c)
# print(idList)
#Zip fucntion used to map id and price_range.
outcome_data = list(zip(idList, predictValues))

#Using DataFrames to prepare the data.
writeData = pd.DataFrame(outcome_data)
writeData.columns = ['id','price_range']

#Writing predicted values to CSV file.
writeData.to_csv("svm_poly.csv", index=False)