#1. import libraries


!pip install numpy==2.2.0
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install xgboost==2.1.3
!pip install pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

#2. load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target #returns X and y as numpy arrays
data.feature_names[:] # all features names. no column name for target var

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# moving data to df for easier visualisation (not that it helps)
X_df = pd.DataFrame({'col1':data.data[:,0], 'col2':data.data[:,1],'col3':data.data[:,2], 'col4':data.data[:,3],'col5':data.data[:,4], 'col6':data.data[:,5],'col7':data.data[:,6], 'col8':data.data[:,7]})
y_df=pd.DataFrame({'target':data.target})
y_df

#no of observations and features
X.shape #(20640 observations, 8 features)
#another way that we can use
N_observations, N_features = X.shape
#print('Number of Observations: ' + str(N_observations))
#print('Number of Features: ' + str(N_features))

#3.develop the models: Random Forest regression and XGBoost regression
n_estimators=100 #define the number of base estimators (individual trees) 

#3.1. Random Forest Model
#intialize the model, fit it, measure the fitting time
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
start_time_rf = time.time() #record the start of fitting time for Random Forest
rf.fit(X_train, y_train) #fit the model
end_time_rf = time.time() #record the end of fitting time for Random Forest
rf_train_time = end_time_rf - start_time_rf #calculate the fitting time for Random Forest
#make prediction, measure prediction time
start_time_yhat_rf=time.time() 
yhat_rf=rf.predict(X_test)
end_time_yhat_rf=time.time() 
rf_predict_time= end_time_yhat_rf - start_time_yhat_rf
#evaluate the model using mse and r2
r2_rf=r2_score(yhat_rf,y_test)
mse_rf=mean_squared_error(yhat_rf,y_test)
print('Random Forest - train time, prediction time, r2, mse: ', rf_train_time, rf_predict_time, r2_rf, mse_rf)
#Random Forest - train time, prediction time, r2, mse:  15.040970802307129 0.16042876243591309 0.7482459593063034 0.255553781221915

#3.2. XGBoost Model
#intialize the model, fit it, measure the fitting time
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)
start_time_xgb = time.time() #record the start of fitting time for XGBoost
xgb.fit(X_train, y_train) #fit the model
end_time_xgb = time.time() #record the end of fitting time for XGBoost
xgb_train_time = end_time_xgb - start_time_xgb #calculate the fitting time for XGBoost
#make prediction, measure prediction time
start_time_yhat_xgb=time.time() 
yhat_xgb=xgb.predict(X_test)
end_time_yhat_xgb=time.time() 
xgb_predict_time= end_time_yhat_xgb - start_time_yhat_xgb
#evaluate the model using mse and r2
r2_xgb=r2_score(yhat_xgb,y_test)
mse_xgb=mean_squared_error(yhat_xgb,y_test)
print('XGBoost - train time, prediction time, r2, mse: ', xgb_train_time, xgb_predict_time, r2_xgb, mse_xgb)
#XGBoost - train time, prediction time, r2, mse:  0.2902028560638428 0.00828099250793457 0.8023960063840978 0.2225899267544737
#XGBoost has slightly better results, but massively shorter computation time


#4. visualise the results
#measure standard deviation of y_test data
std_y = np.std(y_test)

plt.figure(figsize=(14, 6))

# Random Forest plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, yhat_rf, alpha=0.5, color="blue",ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(0,6)
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()


# XGBoost plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, yhat_xgb, alpha=0.5, color="orange",ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(0,6)
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Actual Values")
plt.legend()
plt.tight_layout()
plt.show()

