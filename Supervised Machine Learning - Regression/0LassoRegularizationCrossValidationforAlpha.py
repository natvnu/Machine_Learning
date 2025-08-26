#lasso regularization for feature selection on diabetes disease progression 

#1. install and import libraries
'''
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install seaborn
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
import seaborn as sns

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


#3. load the Dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names
# Create a DataFrame for a clear view
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
#divide x and y
x=df.drop(['target'],axis=1)
y=df[['target']]

#4. develop the model
#train-test split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=42)

#scale x_train and x_test
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test) #transform, do not fit again

#first we will choose alpha using LassoCV which performs cross-validation to find the optimal alpha.
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(x_train_scaled, y_train)
print('Best alpha is: ', lasso_cv.alpha_) #alpha=0.10955050687279648

#develop lasso regressio model with the best alpha from above
lasso=Lasso(alpha=lasso_cv.alpha_)
lasso.fit(x_train_scaled, y_train)
y_pred_lasso = lasso.predict(x_test_scaled)
printmetrics(y_test,y_pred_lasso,lasso) #explained_variance:  0.4809, r2:  0.4782 - Poor

#create coeficients importance dataframe
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Lasso_Coefficient': lasso.coef_
})

#compare "coeficients importance dataframe" and simple "correlaction with target var" dataframe
coef_df #5 out of 10 features are not informative
df.corr()[['target']] # 3 or 4 out of 10 features seem uncorrelated with target var, and some of them ARE not the same as those identified by lasso
#conclusion: Lasso regularization is almost always better for feature selection for predictive modeling. df.corr() is a useful exploratory tool,
#but it has significant limitations.

#Plot predictions vs. actuals
# dataFrame for plotting, combining y_test and y_pred_rf
df_plotting= pd.DataFrame(y_test)
df_plotting['Predicted']=y_pred_lasso
#({'Actual': y_test, 'Predicted': y_pred_lasso})
plt.figure(figsize=(8, 8))
# KDE plot
sns.kdeplot(data=df_plotting, x='target', label='Actual (y_test)', fill=True, alpha=0.3, color='blue')
sns.kdeplot(data=df_plotting, x='Predicted', label='Predicted (y_pred)', fill=True, alpha=0.3, color='red')
plt.xlabel('True Value')
plt.ylabel('Lasso Prediction')
plt.title('True vs. Predicted (Lasso)')
plt.show() #there is a gap between actual and predicted values

#as next steps we could try polynomial transformation of features, or we can abandon linear model and try
#Random Forest or Gradient Boosting (XGBoost) 
