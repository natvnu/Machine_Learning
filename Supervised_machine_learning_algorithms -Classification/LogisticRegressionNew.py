#Using a telecommunications dataset for predicting customer churn

#1. install libraries
!pip install pandas
!pip install scikit-learn
!pip install matplotlib
import pandas as pd
import numpy as numpyp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
%matplotlib inline 

#2. import dataset 
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)
churn_df

#3. data wrangling
churn_df.dtypes #target var churn is float, we need to change it to int, it is requirement by the scikit-learn algorithm
churn_df['churn']=churn_df['churn'].astype('int')
#define x and y
x=churn_df.drop(['churn'],axis=1)
y=churn_df['churn']
#standardize the dataset, it is a norm
x_norm = StandardScaler().fit(x).transform(x) #creates np array
#train-test split the data, 20% for testing
x_train,x_test,y_train, y_test=train_test_split(x_norm,y,test_size=0.2,random_state=4)
#create logistic Regression object
lr=LogisticRegression()
#fit the model
lr.fit(x_train,y_train)
#predict class
yhat=lr.predict(x_test)
yhat[:10]#array of first 10 predictions
#predict probability of belonging to the class
yhat_prob = lr.predict_proba(x_test)
yhat_prob[:10]


#Large positive value of LR Coefficient for a given field indicates that increase in this parameter 
#will lead to better chance of a positive, i.e. 1 class. 

coefficients = pd.Series(lr.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

#calculate results using log_loss metrics
print('Log loss is: ', log_loss(y_test, yhat_prob)) #0.7780756315443422, which is not that good, should be closer to 0
