#REGRESSION PROBLEM
#Random Forest and XGBoost regression models for predicting house prices
#using the California Housing Dataset. 'Performance' means both speed and accuracy.

'''
!pip install numpy==2.2.0
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install xgboost==2.1.3
! pip install pandas
!pip install seaborn
'''
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time


# Load the California Housing dataset
data = fetch_california_housing()
x, y = data.data, data.target


# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# develop models 
n_estimators=100 # we will use the same no of estimators for both 

#random forest - boosting
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
# Measure training time for Random Forest
start_time_rf = time.time()
rf.fit(x_train, y_train)
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf
# Measure prediction time for Random Forest
start_time_rf = time.time()
y_pred_rf = rf.predict(x_test)
end_time_rf = time.time()
rf_pred_time = end_time_rf - start_time_rf
#evaluate Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest:  MSE = {mse_rf:.4f}, R^2 = {r2_rf:.4f}')
print(f'Random Forest:  Training Time = {rf_train_time:.3f} seconds, Testing time = {rf_pred_time:.3f} seconds')


#XGBoost - bagging
# Measure training time for XGBoost
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)
start_time_xgb = time.time()
xgb.fit(x_train, y_train)
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb
# Measure prediction time for Random Forest
start_time_xgb = time.time()
y_pred_xgb = xgb.predict(x_test)
end_time_xgb = time.time()
xgb_pred_time = end_time_xgb - start_time_xgb
#evaluate XGBoost
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f'XGBoost:  MSE = {mse_xgb:.4f}, R^2 = {r2_xgb:.4f}')
print(f'XGBoost:  Training Time = {xgb_train_time:.3f} seconds, Testing time = {xgb_pred_time:.3f} seconds')

#plot actual vs predicted distribution for Random Forest and XGBoost as subplots
plt.figure(figsize=(14, 6))
#Random Forest subplot
# dataFrame for plotting, combining y_test and y_pred_rf
df_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
# KDE plot
plt.subplot(1, 2, 1)
sns.kdeplot(data=df_rf, x='Actual', label='Actual (y_test)', fill=True, alpha=0.3, color='blue')
sns.kdeplot(data=df_rf, x='Predicted', label='Predicted (y_pred)', fill=True, alpha=0.3, color='red')
plt.title('Random Forest: Actual vs. Predicted Values ')
plt.xlabel('Target Value')
plt.legend()
#XGBoost subplot
# dataFrame for plotting, combining y_test and y_pred_rf
df_xgb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_xgb})
# KDE plot
plt.subplot(1, 2, 2)
sns.kdeplot(data=df_xgb, x='Actual', label='Actual (y_test)', fill=True, alpha=0.3, color='blue')
sns.kdeplot(data=df_xgb, x='Predicted', label='Predicted (y_pred)', fill=True, alpha=0.3, color='red')
plt.title('XGBoost: Actual vs. Predicted Values ')
plt.xlabel('Target Value')
plt.legend()
plt.tight_layout()
plt.show()
