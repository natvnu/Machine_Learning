#Multiple Linear Regression, KDE plot, Grid Search for Insurance Charges Prediction

import piplite
await piplite.install('seaborn')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
await download(filepath, "insurance.csv")
file_name="insurance.csv"
df = pd.read_csv(file_name, header=None)
df.columns = ['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region', 'charges']
df.head()
df.replace('?',np.nan, inplace=True)
avg_age=df['age'].astype('float').mean()
df['age']=df['age'].replace(np.nan,avg_age)
mode_smoker=df['smoker'].value_counts().idxmax()
df['smoker']=df['smoker'].replace(np.nan, mode_smoker)
df.isnull().sum()
df['age'] = df['age'].astype('float')
df['charges']=df['charges'].round(2)

df.corr() #smoker is the most important feature, but we will consider them all

#MODEL DEVELOPMENT and refinement
#1. multi-linear regression
y=df['charges']
x=df.drop('charges',axis=1)
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=42)
mlr=LinearRegression()
mlr.fit(x_train,y_train)
y_pred_mlr= mlr.predict(x_test)
print('R2 score of Multiple Linear Regression: ', mlr.score(x_test,y_test)) #R2 score: 0.7395779791474257
print('MSE of Multiple Linear Regression: ',mean_squared_error(y_pred_mlr,y_test)) #MSE: 39969821.42786918
#1.2.display the difference between actual and predicted values
#ax=sns.kdeplot(df['charges'], color='r', label='Actual Values')
#sns.kdeplot(y_pred_mlr, color='b', label='Predicted Values' )
#plt.title('Difference between Actual and Values Predicted by Multiple Lin Reg')
#plt.xlabel('Insurance Charges')
#plt.ylabel('No of Samples')
#plt.legend()
#plt.show() #2 lines, red one for actual values, blue one for Multiple Linear Regression predition

#2.polynomial transformation of features and Grid Search using Ridge as estimator
#2.1. transform features
pr=PolynomialFeatures(2)
x_train_pr=pr.fit_transform(x_train)
x_test_pr=pr.fit_transform(x_test)
x_train_pr.shape #we transformed the number of features from 6 to 28

#2.2. choose hyper parameter values to try out
alpha_param={'alpha':[0.001,0.01,0.1,1,10,100]}
#create RidgeRegression object
RR=Ridge()
#create GridSearchCV object and pass RegressionObject, parameters and no of folds
GS = GridSearchCV(RR, alpha_param, cv=4)
#fit the GridSearch model with train data
GS.fit(x_train_pr,y_train) #if we used x_train here, the result would be the same like in Multiple Linear Regression, as x_train is not polynomialy transformed
#calc the best estimator (Ridge object that contains the optimal alpha)
best_alpha=GS.best_estimator_
all_scores=GS.cv_results_
best_score=best_alpha.score(x_test_pr,y_test)
#print scores
print('Best alpha is:', best_alpha) #the best alpha from above set is 1
print('Best R2 score is:', best_score)#the R2 score for alpha = 1 is 0.8338299261174955
#cast alpha values (dictionary) to a list to be able to use it in a plot
alpha_for_plot=[]
for a in alpha_param.values():
    alpha_for_plot=a
#extract only scores from dictionary all_scores and store them in np array
scores=all_scores['mean_test_score']
#cast np array to list
scores_for_plot = scores.tolist()
'''
#plot the relation between alpha and scores to find where to search for alpha
plt.figure(figsize=(15, 6))
plt.plot(alpha_for_plot,scores_for_plot)
plt.title('Relation between R2 score and alpha')
plt.xlabel('alpha')
plt.ylabel('R2 scores')
plt.show()#R2 sharply declines after around 1, so redo GridSearch with new set of alpha [0.1-1.4])

'''
#2.3. choose different alpha parameter values and redo GridSearch to refine
alpha_param1={'alpha': [0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1,1.2,1.3,1.4]}
#create GridSearchCV object and pass RegressionObject, NEW parameters and no of folds
GS = GridSearchCV(RR, alpha_param1, cv=4)
#fit the GridSearch model with train data
GS.fit(x_train_pr,y_train) 
#calc the best estimator (Ridge object that contains the optimal alpha)
best_alpha=GS.best_estimator_
all_scores=GS.cv_results_
best_score=best_alpha.score(x_test_pr,y_test)
#print scores
print('Best alpha is:', best_alpha) #the best alpha from above set is 0.7
print('Best R2 score is:', best_score)#the R2 score for alpha = 1 is 0.8338696207884985
#cast alpha values (dictionary) to a list to be able to use it in a plot
alpha_for_plot=[]
for a in alpha_param1.values():
    alpha_for_plot=a
#extract only scores from dictionary all_scores and store them in np array
scores=all_scores['mean_test_score']
#cast np array to list
scores_for_plot = scores.tolist()
'''
#plot R2 in relation to alpha
plt.figure(figsize=(15, 6))
plt.plot(alpha_for_plot,scores_for_plot)
plt.title('Relation between R2 score and alpha')
plt.xlabel('alpha')
plt.ylabel('R2 scores')
plt.show()#it is clear from the graph thah alpha around 0.7 is the best choice
'''
#2.4. display the difference between actual and predicted values
#first we need to keep predicted values stored in a var
y_pred_gs = GS.best_estimator_.predict(x_test_pr)
plt.figure(figsize=(15, 6))
ax=sns.kdeplot(y_test, color='r', label='Actual Values' )
sns.kdeplot(y_pred_mlr, color='b', label='Multiple Linear Reg Prediction')
sns.kdeplot(y_pred_gs, color='g', label='Ridge Reg Prediction')
plt.title('Actual Values vs Multiple Lin Reg vs Ridge Reg Prediction')
plt.xlabel('Insurance Charges')
plt.ylabel('No of Samples')
plt.legend()
plt.show()

