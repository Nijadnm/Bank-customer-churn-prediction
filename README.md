#**Overview of what I have done.<br />**		
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
The model achieved an accuracy of 84.5.<br />
<br />
<br />
