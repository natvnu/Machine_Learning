#Implement, evaluate, and compare the performance of three regularization techniques for linear regression
#Analyze the effect of simple linear regularization when modelling on noisy data with and without outliers
#Use Lasso regularization to reduce the number of features for subsequent multiple linear regression modelling

#1.Install and import libraries 

!pip install numpy
!pip install pandas
!pip install scikit-learn
!pip install matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

#2.Define method that prints model evaluation results
def regression_results(y_true, y_pred, regr_type):

    # Regression metrics
    ev = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred) 
    mse = mean_squared_error(y_true, y_pred) 
    r2 = r2_score(y_true, y_pred)
    
    print('Evaluation metrics for ' + regr_type + ' Linear Regression')
    print('explained_variance: ',  round(ev,4)) 
    print('r2: ', round(r2,4))
    print('MAE: ', round(mae,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    print()


******      SIMPLE LINEAR REGRESSION      ******
# 2. Generate synthetic data
noise=1
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + noise*np.random.randn(1000, 1)  # Linear relationship with some noise
y_ideal =  4 + 3 * X
# Specify the portion of the dataset to add outliers (e.g., the last 20%)
y_outlier = pd.Series(y.reshape(-1).copy())

# Identify indices where the feature variable X is greater than a certain threshold
threshold = 1.5  # Example threshold to add outliers for larger feature values
outlier_indices = np.where(X.flatten() > threshold)[0]

# Add outliers at random locations within the specified portion
num_outliers = 5  # Number of outliers to add
selected_indices = np.random.choice(outlier_indices, num_outliers, replace=False)

# Modify the target values at these indices to create outliers (add significant noise)
y_outlier[selected_indices] += np.random.uniform(50, 100, num_outliers)


#3. Plot the data with outliers and the ideal fit line
plt.figure(figsize=(12, 6))
# Scatter plot of the original data with outliers
#THIS IS HOW TO PLOT DOTS AND LINE IN THE SAME PLOT
plt.scatter(X, y_outlier, alpha=0.4,ec='k', label='Original Data with Outliers')
# Line plot of the ideal data, noise free
plt.plot(X, y_ideal,  linewidth=3, color='g',label='Ideal, noise free data')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('')
plt.legend()
plt.show()

#4. Plot the data without the outliers and the ideal fit line
plt.figure(figsize=(12, 6))
# Scatter plot of the original data without outliers
plt.scatter(X, y, alpha=0.4,ec='k', label='Original Data without Outliers')
# Line plot of the ideal data, noise free
plt.plot(X, y_ideal,  linewidth=3, color='g',label='Ideal, noise free data')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('')
plt.legend()
plt.show()

#5. Fit Ordinary, Ridge, and Lasso regression models with outlier data and use them to make predicitions on the original, outlier-free data
# Fit a ordinary linear regression model
lr=LinearRegression().fit(X,y_outlier)
ypred_lr=lr.predict(X)
# Fit a ridge regression model (regularization to control large coefficients)
rr=Ridge(alpha=1).fit(X,y_outlier)
ypred_rr=rr.predict(X)
# Fit a ridge regression model (regularization to control large coefficients)
lasso=Lasso(alpha=.2).fit(X,y_outlier)
ypred_lasso=lasso.predict(X)

#6. Print the regression results
#call function regression_results for Ordinary Linear Regression
regression_results(y, ypred_lr, 'Ordinary Linear Regression') #r2:  0.6357
#call function regression_results for Ridge Regression
regression_results(y, ypred_rr, 'Ridge Regression') #r2:  0.6357
#call function regression_results for Lasso Regression
regression_results(y, ypred_lasso, 'Lasso Regression') #r2:  0.7003

#7.Plot the data and the predictions for comparison - with outlier data
plt.figure(figsize=(12, 6))
# Scatter plot of the original data with outliers
plt.scatter(X, y, alpha=0.4,ec='k', label='Original Data')
# Plot the ideal regression line (noise free data)
plt.plot(X, y_ideal,  linewidth=2, color='k',label='Ideal, noise free data')
# Plot predictions from the ordinary linear regression model
plt.plot(X, ypred_lr,  linewidth=5, label='Ordinary Linear Regression')
# Plot predictions from the ridge regression model
plt.plot(X, ypred_rr, linestyle='--', linewidth=2, label='Ridge Regression')
# Plot predictions from the lasso regression model
plt.plot(X, ypred_lasso,  linewidth=2, label='Lasso Regression')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Comparison of Predictions with Outliers')
plt.legend()
plt.show()
#conclusion: ordinary linear and ridge resgression performed similarly, while Lasso outperformed both. 
#Although the intercept is off for the Lasso fit line, it's slope is close, if not equal to ideal line (it is parallel to ideal line, but a bit off). 
#All three lines were 'pulled up' by the outliers (can't see them here - compare to the plot above where the outliers are shown), with Lasso dampening that effect.

#8. Fit Ordinary, Ridge, and Lasso regression models with data without outliers and use them to make predicitions on the original, outlier-free data
#fit ordinary linear model and make prediction
lr_no=LinearRegression().fit(X,y)
ypred_lr_no=lr_no.predict(X)
#fit ridge regression model and make prediction
rr_no=Ridge(alpha=1).fit(X,y)
ypred_rr_no=rr_no.predict(X)
#fit lasso linear model and make prediction
lasso_no=Lasso(alpha=0.2).fit(X,y)
ypred_lasso_no=lasso_no.predict(X)

#9. Print the regression results
#call function regression_results for Ordinary Linear Regression
regression_results(y, ypred_lr_no, 'Ordinary Linear Regression') #r2:  0.7492
#call function regression_results for Ridge Regression
regression_results(y, ypred_rr_no, 'Ridge Regression')
#call function regression_results for Lasso Regression
regression_results(y, ypred_lasso_no, 'Lasso Regression') #r2:  0.7492

#10.Plot the data and the predictions for comparison - without outlier data
plt.figure(figsize=(12, 6))
plt.scatter(X,y,alpha=0.5, label='Original data')
plt.plot(X,y_ideal, c='black', label='Ideal, noise free data')
plt.plot(X,ypred_lr_no, linewidth=5, label='Ordinary Linear Regression')
plt.plot(X,ypred_rr_no, linewidth=2, linestyle='--', label='Ridge Regression')
plt.plot(X,ypred_lasso_no, linewidth=2, label='Lasso Regression')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Comparison of Predictions without Outliers')
plt.legend()
plt.show()
#conclusion: ordinary linear and ridge resgression performed similarly, with a scope and intercept relatively close to ideal
#they both outperformed Lasso with a slope that was a bit off
'''

#******      MULTIPLE LINEAR REGRESSION      ******
#3. create a high dimensional synthetic dataset with a small number of informative features using make_regression
from sklearn.datasets import make_regression
X, y, ideal_coef = make_regression(n_samples=100, n_features=100, n_informative=10, noise=10, random_state=42, coef=True)

# Get the ideal predictions based on the informative coefficients used in the regression model
ideal_predictions = X @ ideal_coef

# Split the dataset into training and testing sets (?!?tf?)
X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(X, y, ideal_predictions, test_size=0.3, random_state=42)

#4. Fit Ordinary, Ridge, and Lasso regression models with train data and make prediction on test data
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)
linear = LinearRegression()
# Fit the models
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
linear.fit(X_train, y_train)
# Predict on the test set
y_pred_linear = linear.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

'''
#5. Print the regression results
regression_results(y_test, y_pred_linear, 'Ordinary')  #r2:  0.4018
regression_results(y_test, y_pred_ridge, 'Ridge') #r2:  0.4079
regression_results(y_test, y_pred_lasso, 'Lasso') #r2:  0.9815
#conclusion The results for ordinary and ridge regession are poor, explained variances are under 50%, and R^2 is very low. However, the result for Lasso is stellar.


#6. Plot the predictions vs actual
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

#scatter plot between actual and predicted values
axes[0,0].scatter(y_test, y_pred_linear, color="red", label="Linear")
# line drawn between min and max value of actual values
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0,0].set_title("Linear Regression")
axes[0,0].set_xlabel("Actual",)
axes[0,0].set_ylabel("Predicted",)

axes[0,1].scatter(y_test, y_pred_ridge, color="green", label="Ridge")
axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0,1].set_title("Ridge Regression",)
axes[0,1].set_xlabel("Actual",)

axes[0,2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0,2].set_title("Lasso Regression",)
axes[0,2].set_xlabel("Actual",)

# Line plots for predictions compared to actual and predicted values
axes[1,0].plot(y_test, label="Actual", lw=2)
axes[1,0].plot(y_pred_linear, '--', lw=2, color='red', label="Linear")
axes[1,0].set_title("Linear vs Ideal",)
axes[1,0].legend()
 
axes[1,1].plot(y_test, label="Actual", lw=2)
# axes[1,1].plot(ideal_test, '--', label="Ideal", lw=2, color="purple")
axes[1,1].plot(y_pred_ridge, '--', lw=2, color='green', label="Ridge")
axes[1,1].set_title("Ridge vs Ideal",)
axes[1,1].legend()
 
axes[1,2].plot(y_test, label="Actual", lw=2)
axes[1,2].plot(y_pred_lasso, '--', lw=2, color='blue', label="Lasso")
axes[1,2].set_title("Lasso vs Ideal",)
axes[1,2].legend()
 
plt.tight_layout()
plt.show()

'''

#7. Model coefficients
linear_coeff = linear.coef_
ridge_coeff = ridge.coef_
lasso_coeff = lasso.coef_


# 8. Plot the coefficients to see where is the treshhold for
x_axis = np.arange(len(linear_coeff)) #creates array of evenly spaced values. no of values is equal to number of values in linear_coeff, that is 100
x_labels = np.arange(min(x_axis),max(x_axis),10) #creates array of 10 evenly spaced values, between min and max of above
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
plt.show()

#we can see from the plot how much closer the Lasso coefficients are to the ideal coefficients than linear and ridge
#use this plot to choose treshold for features selection => Hint: it will be 5


#9. Use Lasso to select the most important features and compare the three different linear regression models again on the resulting data
#Part 1. Choose a threshold value to select features based on the Lasso model coefficients => 5
##Create a dataframe to compare the Lasso coefficients with the ideal coefficients. It will contain all ideal coeficients,
#all lasso coeficients and one extra col which will have True value if Lasso coef > treshold (if feature is important accoring to Lasso)

treshold=5
#create a df with all Ideal coef values
feature_importance_df = pd.DataFrame(ideal_coef.T, columns=['Ideal Features'])
#add all Lasso_coeff values
feature_importance_df['Lasso Features']=pd.DataFrame(lasso_coeff.T)
#create new column that is True if chosen by Lasso, aka if Lasso_coeff is > than treshold, otherwise it is False
feature_importance_df['Feature Selected'] = feature_importance_df['Lasso Features'].abs() > treshold
print('Ideal, Non-zero features') #where Ideal_coef is >0
display(feature_importance_df.loc[feature_importance_df['Ideal Features'].abs()>0])#4,5,6,36,50,54,66,68,78,82,87
print('Lasso selected features') #where lasso_coeff is >treshold
display(feature_importance_df.loc[feature_importance_df['Feature Selected']==True])#4,5,6,50,54,66,68,78,82,87 #lasso chose 9 out of 10 ideal features

#Conclusion The result is very good. We managed to correctly identify 9 out of the 10 important features.
#We cannot use ideal_coef for feature selection, because in real-world datasets, the "ideal" features are not known in advance. Here we have them for purpose of excersize

#Part 2. Use the threshold to select the most important features and then create a new model
important_features = feature_importance_df[feature_importance_df['Feature Selected']].index #store indexes of the features selected by Lasso
#Keep important features in X
X_filtered = X[:, important_features] #np array of 100 observations and 10 features - index + 9 features selected by Lasso(instead of original 100 features)
print("Shape of the filtered feature set:", X_filtered.shape)
# Split the dataset into training and testing sets again
X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(X_filtered, y, ideal_predictions, test_size=0.3, random_state=42)

# Initialize the models
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)
linear = LinearRegression()

# Fit the models
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
linear.fit(X_train, y_train)

# Predict on the test set
y_pred_linear = linear.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

#evaluate the models and print out the results 
regression_results(y_test, y_pred_lasso, 'Lasso') #new result => r2:  0.9914 is a bit of improvement compared to previous 0.9815
regression_results(y_test, y_pred_ridge, 'Ridge') #new result => r2:  0.9905 is a drastic improvement compared to previous 0.4079
regression_results(y_test, y_pred_linear, 'Ordinal') #new result => r2:  0.9915 is a drastic improvement compared to previous 0.4012
#conclusion: The new results are vastly improved for ordinary and Ridge regression, and slightly improved for Lasso, 
#supporting the idea that Lasso regression can be very beneficial when used as a feature selector.

# 10. Plot the predictions vs actuals for new X_filtered
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

axes[0,0].scatter(y_test, y_pred_linear, color="red", label="Linear")
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0,0].set_title("Linear Regression",)
axes[0,0].set_xlabel("Actual",)

axes[0,2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0,2].set_title("Lasso Regression",)
axes[0,2].set_xlabel("Actual",)
axes[0,2].set_ylabel("Predicted",)

axes[0,1].scatter(y_test, y_pred_ridge, color="green", label="Ridge")
axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0,1].set_title("Ridge Regression",)
axes[0,1].set_xlabel("Actual",)

axes[0,2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0,2].set_title("Lasso Regression",)
axes[0,2].set_xlabel("Actual",)
axes[0,2].set_ylabel("Predicted",)

# Line plots for predictions compared to actual and ideal predictions
axes[1,0].plot(y_test, label="Actual", lw=2)
axes[1,0].plot(y_pred_linear, '--', lw=2, color='red', label="Linear")
axes[1,0].set_title("Linear vs Ideal",)
axes[1,0].legend()
 
axes[1,1].plot(y_test, label="Actual", lw=2)
axes[1,1].plot(y_pred_ridge, '--', lw=2, color='green', label="Ridge")
axes[1,1].set_title("Ridge vs Ideal",)
axes[1,1].legend()
 
axes[1,2].plot(y_test, label="Actual", lw=2)
axes[1,2].plot(y_pred_lasso, '--', lw=2, color='blue', label="Lasso")
axes[1,2].set_title("Lasso vs Ideal",)
axes[1,2].legend()

plt.tight_layout()
plt.show()


