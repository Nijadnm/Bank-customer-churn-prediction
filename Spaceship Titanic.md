**<h1>Overview of Project.</h1><br />**		
After converting the the datdaset into a dataframe using pandas , I had checked for missing values<br />
The dataset containes the columns : PassengerId,	HomePlanet,	CryoSleep,	Cabin,	Destination,	Age,	VI,	RoomService,	FoodCourt,	ShoppingMall,	Spa,	VRDeck,	Name,	Transported.<br />
<br />
After going through the columns I concluded that the columns VI,	RoomService,	FoodCourt,	ShoppingMall,	Spa,	VRDeck,	Name had nothing to do with the modelling and droppd those columns with drop() function.<br />
Both the categorical and numerical columns hasd missing values.<br /><br />
After going through the columns , I had decided to drop the rows in which the categorical vaues were missing.<br />
Used the drop() function to do it.<br />
'PassengerId' contains to important data postion of the person and group of the person.So I split the PassengerId with str.split() function and created to two new column 'pos' and 'group'.<br />
Also the column 'Cabin' conatained important data such as the deck and side where the perons cabin was placed. using str.split() CAbin was split into new columns 'Deck' and 'side'.<br />
target variable: Transported<br />
Features: 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'Transported','Deck', 'Side', 'pos', 'group'<br />
split the dataset inro training and validation set with test size= 0.2.<br />
Two new list were in which each containg the names of columns with categorical and numerical data seperately.<br />
Imputation was done to fill the mssing values.<br />
One HOt Encoding was done on the categorical data.<br />
Used Logistic Regression to train the data with traing dataset<br />
Used Accuracy_score to get the accuracy of the model<br />
Attained an accuracy of **71.9**<br />






**<h1>Code</h1><br />**	
from sklearn.model_selection import train_test_split<br />
from sklearn.preprocessing import OrdinalEncoder<br />
from sklearn.impute import SimpleImputer<br />
from sklearn.preprocessing import OneHotEncoder<br />
from sklearn.ensemble import RandomForestRegressor<br />
from sklearn.metrics import accuracy_score<br />
from sklearn.preprocessing import LabelEncoder<br />
import matplotlib.pyplot as plt<br />
from sklearn.linear_model import LogisticRegression<br />


X=pd.read_csv('../input/spaceship-titanic/train.csv')<br />
X.isnull().sum()<br />
X.drop(['Name','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'],axis=1,inplace=True)<br />

### Getting only the deck by splitting the cabin<br />
new=X['Cabin'].str.split('/',n=2,expand=True)<br />
X['Deck']=new[0]<br />
X['Side']=new[2]<br />
X.drop(['Cabin'],axis=1,inplace=True)<br />


### Getting the persons position in the group<br />
pos=X['PassengerId'].str.split('_',n=1,expand=True)<br />
X['pos']=pos[1]<br />
X['group']=pos[0]<br />
X.drop(['PassengerId',],axis=1,inplace=True)<br />
X = X.astype({'pos':'int'})<br />
X = X.astype({'group':'int'})<br />

object_cols=[i for i in X.columns if X[i].dtype=='object' and X[i].nunique()<10]<br />
for i in object_cols:<br />
    X.dropna(axis=0,subset=[i],inplace=True)<br />

X.dropna(axis=0,subset=['Transported'],inplace=True)<br />
y=X.Transported<br />
X.drop(['Transported'],axis=1,inplace=True)    <br />
labelencoder=LabelEncoder()<br />
y=labelencoder.fit_transform(y)<br />

num_cols=[i for i in X.columns if X[i].dtype in ['int64','float64']]<br />

xtrain,xvalid,ytrain,yvalid=train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=0)<br />
num_transformer=SimpleImputer()<br />
imputed_X_train=pd.DataFrame(num_transformer.fit_transform(xtrain[num_cols]))<br />
imputed_X_valid=pd.DataFrame(num_transformer.transform(xvalid[num_cols]))<br />
imputed_X_train.columns=xtrain[num_cols].columns<br />
imputed_X_valid.columns=xvalid[num_cols].columns<br />



OHE=OneHotEncoder(sparse=False)<br />
OHE_xtrain=pd.DataFrame(OHE.fit_transform(xtrain[object_cols]))<br />
OHE_xvalid=pd.DataFrame(OHE.transform(xvalid[object_cols]))<br />

X_train=pd.concat([imputed_X_train,OHE_xtrain],axis=1)<br />
X_valid=pd.concat([imputed_X_valid,OHE_xvalid],axis=1)<br />

model=LogisticRegression(random_state=0)<br />


model.fit(X_train,ytrain)<br />
preds=model.predict(X_valid)<br />
print(preds)<br />

accuracy = accuracy_score(yvalid, preds)<br />
print(accuracy)<br />
