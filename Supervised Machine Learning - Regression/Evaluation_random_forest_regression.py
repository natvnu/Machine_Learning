#Use the California Housing data set included in scikit-learn to predict the median house price based on various attributes
#Create a random forest regression model and evaluate its performance
#Investigate the feature importances for the model
#This dataset was obtained from the StatLib repository.
#https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

#1. install and import libraries

!pip install numpy 
!pip install pandas
!pip install scikit-learn
!pip install matplotlib
!pip install scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error, mean_absolute_error, r2_score

#2.load the dataset
data=fetch_california_housing()
x,y=data.data, data.target #prices are not in USD, if we want them in USD, need to multiply by 10,000

'''
data.feature_names #features columns names
data.target_names #target column names
x_df=pd.DataFrame(x)
x_df.columns=data.feature_names
x_df.head()
y_df=pd.DataFrame(y)
y_df.columns=data.target_names
y_df.head()
'''
#print(data.DESCR) #dataset description

#3. train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#4.develop Random Forest Regression Model
rf=RandomForestRegressor(n_estimators=100,random_state=42) #n_estimators is veryb important
rf.fit(x_train,y_train)
yhat_rf=rf.predict(x_test)

#5. measure MAE, MSE, RMSE and R2
mae = mean_absolute_error(y_test, yhat_rf)
mse = mean_squared_error(y_test, yhat_rf)
rmse = root_mean_squared_error(y_test, yhat_rf)
r2 = r2_score(y_test, yhat_rf)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

#the R2 score is not particularly high, 0.8,
#but is means 80% of the variance in median house prices can be explained the model
#The mean absolute error is $33,220. So, on average, predicted median house prices are off by $33k.
#These statistics alone don't explain any details about the performance of the model: where did the model do well or poorly

'''
#STEPS 6, 7, 8 and 9 are not MANDATORY, these are just to help us understand the results better
#6.Plot Actual vs Predicted values #we see correlation and nothing else
plt.scatter(y_test, yhat_rf)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest Regression - Actual vs Predicted')
plt.show()

#7. Plot the histogram of the residual errors (dollars)
residual_errors=1e5*(yhat_rf-y_test) #1e5=10,000
plt.hist(residual_errors, bins=30,color='lightblue', edgecolor='black')
plt.title('Median House Value Prediction Residual Errors')
plt.xlabel('Median House Value Prediction Residual Error ($)')
plt.ylabel('Frequency')
plt.show()
print('Average error = ', np.mean(residual_errors))
print('Standard deviation of error = ', np.std(residual_errors))
#The residuals are normally distributed with a very small average error and a standard deviation of $50,518.
#Standard deviation is a statistical measure that quantifies the amount of variation or dispersion in a 
#set of values. It indicates how much the individual data points deviate from the mean (average) of the 
#data set. A low standard deviation means that the data points are close to the mean, 
#while a high standard deviation indicates that the data points are spread out over a wider range.
#The standard deviation is calculated using the following steps:
#Calculate the mean of the data set.
#Subtract the mean from each data point and square the result.
#Find the mean of these squared differences.
#Take the square root of this mean to obtain the standard deviation.
#Formula for Standard Deviation: σ = √(Σ (Xi - μ)² / N)
#The standard deviation (σ) is calculated as: σ = √(Σ (Xi - μ)² / N), 
#where: σ = Standard deviation, N = Number of observations, Xi = ith observation, μ = Mean

#8. Plot the distribution plot of predicted vs actual values
ax1=sns.kdeplot(y_test, color='r', label='Actual Value')
sns.kdeplot(yhat_rf, color='b', label='Predicted Values')
plt.title('Actual vs predicted values of the house')
plt.xlabel('Price')
plt.ylabel('No of samples')
plt.legend(['Actual prices','Predicted prices'])
plt.show()
'''
#9. Display feature importances as barchart
# Feature importances
importances = rf.feature_importances_
fi_df=pd.DataFrame(importances)
fi_df.columns=['Importances']
fi_df['Features']=pd.DataFrame(data.feature_names)
fi_df=fi_df.sort_values(by='Importances')

fi_df.plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks([3,4,2,1,7,6,5,0],fi_df['Features'])
plt.title("Feature Importances in Random Forest Regression")
plt.show()
