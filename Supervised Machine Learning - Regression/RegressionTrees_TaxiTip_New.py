'''
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
'''
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import normalize, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# import data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
raw_data.isnull().sum()
raw_data.corr()[['tip_amount']].sort_values(by='tip_amount') #features are not correlated with the target var
#display the correlation with target var on horizontal bar
#correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
#correlation_values.plot(kind='barh', figsize=(10, 6))

#develop model
y=raw_data['tip_amount']
x=raw_data.drop(['tip_amount'], axis=1)
#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#create DecisionTreeRegressor object
dt_reg = DecisionTreeRegressor(criterion = 'poisson', max_depth=4, random_state=35)
#fit the model
dt_reg.fit(x_train, y_train)
#make prediction
y_pred = dt_reg.predict(x_test)
#evaluate
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))#MSE score for square_error criterion, max_depth=8: 25.198
r2_score = dt_reg.score(x_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))#R^2 score square_error criterion, max_depth=8: 0.002

#refine/redevelop the model
#go back to criterion and try poisson, absolute_error or friedman_mse, change depth =>
#=> poisson with max_depth=4 returns the best result, R^2 score : 0.043
#reselect the features - remove the ones with lowest correlation
x=raw_data.drop(['tip_amount','payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis=1)
#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#create DecisionTreeRegressor object
dt_reg = DecisionTreeRegressor(criterion = 'squared_error', max_depth=4, random_state=35)
#fit the model
dt_reg.fit(x_train, y_train)
#make prediction
y_pred = dt_reg.predict(x_test)
#evaluate
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))#MSE score for squared_error criterion, max_depth=4: 24.122
r2_score = dt_reg.score(x_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))#R^2 score squared_error criterion, max_depth=4: 0.045
#the model still performs poorly
