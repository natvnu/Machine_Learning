#1. import libraries
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


import warnings
warnings.filterwarnings('ignore')

#2.import dataset
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
raw_data

#3.explore the dataset
#plot the correlation between features and target var
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
#correlation_values.plot(kind='barh', figsize=(10, 6))
#or simply print out the corr table
raw_data.corr()

#4. develop the model
x=raw_data.drop('tip_amount',axis=1)
y=raw_data['tip_amount']
#normalize x data
x = normalize(x, axis=1, norm='l1', copy=False)
#train-test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
#create a regression tree object
rt=DecisionTreeRegressor(criterion = 'squared_error', max_depth=8, random_state=35)
#fit the data
rt.fit(x_train,y_train)
#make prediction
yhat_rt=rt.predict(x_test)
yhat_rt
#evaluate the model
mse_score = mean_squared_error(y_test, yhat_rt)
print('Regression Tree depth 8 MSE score : ', mse_score) #24.55457832664192, depth 12 will yield 26.498215477547053 
r2_score = rt.score(x_test,y_test)
print('Regression Tree depth 8 R^2 score :', r2_score) #0.027901599689406087, depth 12 will yield -0.049045621315366716
#we can see that the model performs poorly, let's get rid of features that are not relevant

#5. redevelop the model
corr_table=raw_data.corr()
abs(corr_table).sort_values(by='tip_amount',ascending=False)
raw_data=raw_data.drop(columns=['payment_type','VendorID','store_and_fwd_flag','improvement_surcharge'], axis=1)
#define x and y again
x=raw_data.drop('tip_amount',axis=1)
y=raw_data['tip_amount']
#normalize x data again
x = normalize(x, axis=1, norm='l1', copy=False)
#train-test split again
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
#create a regression tree object again, but this time with max_depth = 4
rt=DecisionTreeRegressor(criterion = 'squared_error', max_depth=4, random_state=35)
#fit the data again
rt.fit(x_train,y_train)
#make prediction again
yhat_rt=rt.predict(x_test)
#evaluate the model again
mse_score = mean_squared_error(y_test, yhat_rt)
print('Regression Tree depth 4 MSE score : ', mse_score) #24.468134714996847
r2_score = rt.score(x_test,y_test)
print('Regression Tree depth 4 R^2 score :', r2_score) #0.031323841174452194
#the model still performs poorly
