#Using a telecommunications dataset for predicting customer churn

#Using a telecommunications dataset for predicting customer churn
'''
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

'''


#2. import dataset 
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)
churn_df

#3. data wrangling
churn_df.dtypes #target var churn is float, we need to change it to int, it is requirement by the scikit-learn algorithm
churn_df['churn']=churn_df['churn'].astype('int')

# 4. classifier development
#define x and y
#first round 
#x=churn_df.drop(['churn'],axis=1) 
#second round
x = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ',  'callcard','churn']]
y=churn_df['churn']
#scale the dataset
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
#predict probability of non churn vs churn (0,1)
yhat_prob = lr.predict_proba(x_test)
yhat_prob[:10]
#calculate results using log_loss metrics
print('Log loss is: ', log_loss(y_test, yhat_prob)) #0.7780756315443422, which is not that good, should be closer to 0
#going back to features and choosing 'tenure', 'age', 'address', 'income', 'ed', 'employ',  'callcard' will reduce log loss to 0.02065304337548963

#to display distribution of predicted probability of customer churn
plt.hist(y_pred_logr[:, 1], bins=20)  # Show distribution of churn probabilities
plt.xlabel('Predicted Probability of Churn')
plt.ylabel('Count')
plt.show()


