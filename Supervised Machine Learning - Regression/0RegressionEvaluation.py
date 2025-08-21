#REGRESSION EVALUATION TECHNIQUES 
#Random Forest for predicting house prices on California Housing Dataset. 

#1. install and import dataset
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
import time

#2. import dataset
# Load the California Housing dataset
data = fetch_california_housing()
x, y = data.data, data.target

#3. develop random forest boosting model

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
n_estimators=100 # we will use the same no of estimators for both 
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
#fit
rf.fit(x_train, y_train)
#predict
y_pred_rf = rf.predict(x_test)
#evaluate 
mae = mean_absolute_error(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)
rmse = root_mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
#R² Score (R-squared): 0.8050 means that the proportion of the variance in the target variable (house price) explained by features is 80.5%
#MAE of 0.3276 means our prediction is, on average, about $32,760 away from the actual price (unit for dataset is 100,000 of $)
#RMSE is 0.5055, which translates to roughly $50,550. The fact that the RMSE is significantly higher than the MAE tells us that the model is making some large errors on a subset of the data. 
#These could be very expensive houses, very cheap houses, or houses with unusual features that are hard for the model to price correctly.

'''
#4. plot actual vs predicted distribution for Random Forest
#y_pred_rf = rf.predict(x_test) - 0.15
plt.figure(figsize=(14, 6))
#Random Forest subplot
# dataFrame for plotting, combining y_test and y_pred_rf
df_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
# KDE plot
#plt.subplot(1, 2, 1)
sns.kdeplot(data=df_rf, x='Actual', label='Actual (y_test)', fill=True, alpha=0.3, color='blue')
sns.kdeplot(data=df_rf, x='Predicted', label='Predicted (y_pred)', fill=True, alpha=0.3, color='red')
plt.title('Random Forest: Actual vs. Predicted Values ')
plt.xlabel('Target Value')
plt.xticks(range(5), ['100k', '200k', '300k', '400k', '500k'])
plt.legend()
plt.show()
#the plot shows that predicted prices are a bit "shifted to the right" when compared to actual ones
#if we reduce the predicted prices for around 15,000 USD they may better fit the reality.
#we can do this by uncomenting the line that reduces the y_pred_rf for 0.15
'''

'''
#5. residual plot - predicted values vs residuals 
#calc residuals - do not forget to multiply with 100,000$
residuals = 1e5*(y_test - y_pred_rf) 
# plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, residuals, alpha=0.5)  # alpha makes points slightly transparent
plt.xlabel('Predicted House Values (hundreds of thousands of dollars)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot for Random Forest Model')
plt.tight_layout()
plt.show()
#There are no curvatures, but there is a there is straight line of dots that start at around x=2 and y=300000 and ends at around x=5 and y=20000
#That straight line means that our data are clipped. All houses that sold for $500,000 or more are recorded in the dataset as simply $500,001.
#For a house whose actual value is capped at 5.0 (500,001$), if the model predicts 5.2, the residual is 5.0 - 5.2 = -0.2.
#If the model predicts 6.0, the residual is 5.0 - 6.0 = -1.0
#This creates a perfect, linear, downward-sloping line: Residual = (Capped Value) - (Predicted Value).
#Our model is actually correctly identifying that these are high-value properties and trying to assign them true values greater than $500,000. 
#But data are limited to 500,001 and therefore the plot reports the error.
#If data were not clipped, the actual value for 5.2 prediction would be closer to 5.2 (maybe would be 5.22 or so)
#Therefore our residual (error) would be lower (5.22-5.2)
#This is a primary reason your RMSE is significantly higher than your MAE. The model is being penalized for correctly understanding that a house is more expensive than the dataset's artificial limit
'''

#6. feature importances
importances = rf.feature_importances_
features = data.feature_names
#create dataframe
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance')
# Plot feature importances
feature_importance_df.plot(kind='bar')
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(range(8), features)
plt.title("Feature Importances in Random Forest Regression")
plt.show()

#the most important feature is longitude
