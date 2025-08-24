
# Regularization in Linear Regression on a generated dataset with a small number of informative features
#Ordinal, Lasso and Ridge Regression. 
#Ploting all features and their importances for all 3 models and comparing their magnitudes with ideal coeficients
#Deciding on treshold for feature imporance and using Lasso regularization to reduce the number of features
#Multiple linear regression modelling (ordinal, Lasso and Ridge Regression) using only important features


#1. install and import libraries

'''
!pip install numpy
!pip install pandas
!pip install scikit-learn
!pip install matplotlib
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

#2. define method that prints metrics
def printmetrics(yt,yp,m):
    ev=explained_variance_score(yt, yp)
    mae = mean_absolute_error(yt, yp) 
    mse = mean_squared_error(yt, yp) 
    r2 = r2_score(yt, yp)
    print('Evaluation metrics for ', m)
    print('explained_variance: ',  round(ev,4)) 
    print('r2: ', round(r2,4))
    print('MAE: ', round(mae,4))
    print('MSE: ', round(mse,4))
    #print('RMSE: ', round(np.sqrt(mse),4))
    print()

    
#3. Create a high dimensional synthetic dataset with a small number of informative features using make_regression
from sklearn.datasets import make_regression

X, y, ideal_coef = make_regression(n_samples=100, n_features=100, n_informative=10, noise=10, random_state=42, coef=True)

# Get the ideal predictions based on the informative coefficients used in the regression model
ideal_predictions = X @ ideal_coef

#4. develop Ordinal, Lasso and Ridge Linear Regression models
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(X, y, ideal_predictions, test_size=0.3, random_state=42)

lasso=Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
printmetrics(y_test,y_pred_lasso,lasso) #explained_variance:  0.9815, r2:  0.9815 - Great

ridge=Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
printmetrics(y_test,y_pred_ridge,ridge) #explained_variance:  0.4446, r2:  0.4079 - Poor

linear=LinearRegression()
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)
printmetrics(y_test,y_pred_linear,linear) #explained_variance:  0.4354, r2:  0.4018 - Poor

#Lasso regression performs really good, while ordinal and ridge regression (even if we change alpha) perform poorly.
#The reason for this is because lasso identified and kept only the most important features (all other features are assigned 0 hyperparameter).
#We can print out feature importances using lasso.coef_, but since we generated the dataset, we have no column names
#meaning that we can only see feature importances, but they will not be assigned to features.
#Therefore we will plot all features and their importances for all 3 models and compare their magnitudes with ideal coeficients

#5. Model coefficients
linear_coeff = linear.coef_
ridge_coeff = ridge.coef_
lasso_coeff = lasso.coef_

# Plot the coefficients
x_axis = np.arange(len(linear_coeff))
x_labels = np.arange(min(x_axis),max(x_axis),10)
plt.figure(figsize=(12, 6))

plt.scatter(x_axis, ideal_coef,  label='Ideal', color='blue', ec='k', alpha=0.4)
plt.bar(x_axis - 0.25, linear_coeff, width=0.25, label='Linear Regression', color='blue')
plt.bar(x_axis, ridge_coeff, width=0.25, label='Ridge Regression', color='green')
plt.bar(x_axis + 0.25, lasso_coeff, width=0.25, label='Lasso Regression', color='red')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Model Coefficients')
plt.xticks(x_labels)
plt.legend()
plt.show() #we can see that lasso-selected features are closest to ideal ones

#6. Use Lasso to select the most important features and compare the three different linear regression models developed using only selected features

threshold = 5 # selected by inspection of coefficients plot

# Create a dataframe containing the Lasso model and ideal coefficients
fi_df = pd.DataFrame(lasso.coef_, columns=['Lasso Coefficients'])
fi_df['Ideal Coefficients']= pd.DataFrame(ideal_coef)

# Add 'Features Selected' column (True/False)
fi_df['Features Selected'] = fi_df['Lasso Coefficients'].abs() > threshold

print("Features Identified as Important by Lasso:")
display(fi_df[fi_df['Features Selected']])

print("\nNonzero Ideal Coefficient Indices")
display(fi_df[fi_df['Ideal Coefficients']>0])

#Lasso identified 9 out of 10 ideal features

#7. Use Lasso-identified features to redevelop the model
#create x_df with only important features
#but first extract names of important features from  fi_df
important_columns= fi_df[fi_df['Features Selected']].index
column_names
x_df=pd.DataFrame(X)
x_df=x_df[important_columns]
#now use x_df to redevelop regression models
X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(x_df, y, ideal_predictions, test_size=0.3, random_state=42)
lasso=Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
printmetrics(y_test,y_pred_lasso,lasso) #explained_variance:  0.9917, r2:  0.9914 - Exceptional

ridge=Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
printmetrics(y_test,y_pred_ridge,ridge) #eexplained_variance:  0.9917, r2:  0.9914 - Exceptional

linear=LinearRegression()
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)
printmetrics(y_test,y_pred_linear,linear) #explained_variance:  0.9917, r2:  0.9915 - Exceptional
#Conclusion: With preselected features all regression models perform exceptionally

