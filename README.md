Overview of what I have done
Firstly I have created a dataframe of the required dataset using pandas.
Then as a part of preprocessing, I checked for any missing values in the dataset.
Fortunately, there was none.
The dataset had columns :'customer_id','credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'estimated_salary','churn'.
Here churn is the target variable
Next I used crosstab and mean to find appropriate features for training the model.
From the analysis, I concluded that almost all the columns contributed to the model.
Then, splitted the data into training and validation set with test size of 0.2
For the categorical data, I did One Hot Encoding using get_dummies() function.
I used XGB classifier model for training the data with training set.
Achieved an accuracy of 84.5
