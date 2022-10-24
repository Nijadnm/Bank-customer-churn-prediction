import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb



X=pd.read_csv("../input/bank-customer-churn-dataset/Bank Customer Churn Prediction.csv")
file=X
X.info()
X.isnull().sum()
y=X.churn
plt.figure(figsize=(10,15))
pd.crosstab(X.country,X.churn).plot(kind="bar")
pd.crosstab(X.gender,X.churn).plot(kind="bar")
pd.crosstab(X.credit_card,X.churn).plot(kind="bar")
plt.scatter(X.tenure,y)
plt.show()
print(X.groupby(X.churn).mean())
X.drop(['customer_id','churn'],axis=1,inplace=True)
labelencoder=LabelEncoder()
X.gender=labelencoder.fit_transform(X.gender)
X.country=labelencoder.fit_transform(X.country)
xtrain,xvalid,ytrain,yvalid=train_test_split(X,y,test_size=0.2,random_state=0)
model=xgb.XGBClassifier(n_estimators=500,random_state=0)
model.fit(xtrain,ytrain)
preds=model.predict(xvalid)

accuracy = accuracy_score(yvalid, preds)
print('**',(accuracy))

