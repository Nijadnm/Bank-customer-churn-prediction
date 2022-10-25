**<h1>Overview of Project.</h1><br />**		
The aim of the project is to predict whether to predict the customer will leave the bank or not.<br />
Firstly I have created a dataframe of the required dataset using pandas.<br />
Then as a part of preprocessing, I checked for any missing values in the dataset.<br />
Fortunately, there was none.<br />
The dataset had columns :'customer_id','credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'estimated_salary','churn'.<br />
Here churn is the target variable.<br />
Next I used crosstab and mean to find appropriate features for training the model.<br />
From the analysis, I concluded that almost all the columns contributed to the model.<br />
Then, splitted the data into training and validation set with test size of 0.2.<br />
For the categorical data, I did Label Encoding for the features ‘country’ and ‘gender’.<br />
I used XGB classifier model for training the data with training set.<br />
The model achieved an accuracy of **84.5.**<br />
<br />
<br />


**<h1>Code</h1><br />**		
import numpy as np <br />
import pandas as pd <br />
import matplotlib.pyplot as plt<br />
from sklearn.preprocessing import LabelEncoder<br />
from sklearn.model_selection import train_test_split<br />
from sklearn.linear_model import LogisticRegression<br />
from sklearn.metrics import accuracy_score<br />

X=pd.read_csv("../input/bank-customer-churn-dataset/Bank Customer Churn Prediction.csv")<br />
X.info()<br />
X.isnull().sum()<br />
y=X.churn<br />
plt.figure(figsize=(10,15))<br />
pd.crosstab(X.country,X.churn).plot(kind="bar")<br />
pd.crosstab(X.gender,X.churn).plot(kind="bar")<br />
pd.crosstab(X.credit_card,X.churn).plot(kind="bar")<br />
plt.scatter(X.tenure,y)<br />
plt.show()<br />
print(X.groupby(X.churn).mean())<br />
X.drop(['customer_id','active_member','churn'],axis=1,inplace=True)<br />
labelencoder=LabelEncoder()<br />
X.gender=labelencoder.fit_transform(X.gender)<br />
X.country=labelencoder.fit_transform(X.country)<br />
xtrain,xvalid,ytrain,yvalid=train_test_split(X,y,test_size=0.2,random_state=0)<br />
model=LogisticRegression(random_state=0)<br />
model.fit(xtrain,ytrain)<br />
preds=model.predict(xvalid)<br />

accuracy = accuracy_score(yvalid, preds)<br />
print('**',(accuracy))<br />


