
#a telecommunications provider has segmented its customer base by service usage patterns, 
#categorizing the customers into four groups. It is a classification problem. 

#1.import libraries
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install seaborn==0.13.2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
%matplotlib inline

#2. read dataset into df
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()
abs(df.corr()).sort_values(by='custcat', ascending= False) #ed, tenure, income, employ have some correlation

#3. develop KNN model
#define x and y (features and target var)
x=df.drop('custcat',axis=1)
y=df['custcat']
#standardize data
x_norm = StandardScaler().fit_transform(x)
#train-test split
x_train,x_test,y_train,y_test=train_test_split(x_norm,y,test_size=0.2, random_state=4)
'''
#create KNN classsifier object
k=3 #number of neighbors we will analyze
knn=KNeighborsClassifier(n_neighbors=k)
#train the model
knn.fit(x_train, y_train)
#make prediction
yhat_knn=knn.predict(x_test)
#evaluate the accuracy of the model
print("KNN model (k=3) accuracy: ", accuracy_score(y_test, yhat_knn)) #0.315
'''
#choice of k affects the model, so we will choose the optimal k
ks=500
# 2 arrays to keep the scores
acc_scores=np.zeros((ks-1))
for n in range(1,ks):
    #train model and predict
    neigh=KNeighborsClassifier(n_neighbors=n)
    neigh.fit(x_train,y_train)
    yhat_neigh=neigh.predict(x_test)
    acc_scores[n-1] = accuracy_score(y_test, yhat_neigh)
acc_scores #array([0.3  , 0.29 , 0.315, 0.32 , 0.315, 0.31 , 0.335, 0.325, 0.34 ]), k=9 is the optimal k
print("The best accuracy was ", acc_scores.max(), "with k =", acc_scores.argmax()+1) #the best accuracy was  0.34 with k = 9
#we can increase k, to see how the model accuracy is changing. The most we will get is 0.41 with k=38
#this is still not very good, so we can try removing weakly correlated features
