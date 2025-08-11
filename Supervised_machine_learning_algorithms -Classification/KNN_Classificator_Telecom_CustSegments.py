
#K-Nearest neighbors to classify data
#A telecommunications provider has segmented its customer base by service usage patterns, 
#categorizing the customers into four groups:
#Basic Service
#E-Service
#Plus Service
#Total Service
#The objective is to build a classifier to predict the service category for unknown cases. 
'''
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install seaborn==0.13.2
'''
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
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()
df.isnull().sum()
df.dtypes
df['custcat'].value_counts()


#illustrate correlation of features with target var
corr_df = df.corr()['custcat'].drop('custcat').sort_values()
'''
plt.figure(figsize=(10, 8))
corr_df.plot(kind='barh')
'''

'''
#prepare x and y
x = df.drop('custcat',axis=1)
y = df['custcat']
#Scaling of data is important for the KNN model, it transforms the data to have mean=0 and standard dev=1
#KNN makes predictions based on the distance between data points (samples) 
#we want all features on the same scale (with no feature dominating due to its larger range).
#This helps KNN make better decisions based on the actual relationships between features, not just on the magnitude of their values.
#scale the features
x=StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

#develop KNN model
k = 3
#Train Model and Predict  
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(x_train,y_train)
y_pred=knn_model.predict(x_test)
print("Test set Accuracy: ", accuracy_score(y_test, y_pred)) #0.315

#plot the model accuracy to identify the model with the most suited value of k.
Ks = 100
acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    y_pred = knn_model_n.predict(x_test)
    acc[n-1] = accuracy_score(y_test, y_pred)
    print('Accuracy: ', acc[n-1], 'for k: ', (n)) #Accuracy:  0.41 for k:  38

#illustrate accuracy in relation to k
plt.plot(range(1,Ks+1),acc,'g')
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()# highest accuracy is achieved with k between 35 and 45, more than 0.4, but not clearn which value from the graph - we can see it is 0.41 for k=38 from the above loop 
'''
# train the model again with selected features and with k between 35 and 45
x = df[['tenure','ed', 'income', 'employ', 'marital','reside']]
y = df['custcat']
x=StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

#plot the model accuracy to identify the model with the most suited value of k.
Ks = 45
acc = np.zeros((Ks))
for n in range(35,Ks+1):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    y_pred = knn_model_n.predict(x_test)
    acc[n-1] = accuracy_score(y_test, y_pred)
    print('Accuracy: ', acc[n-1], 'for k: ', (n))


plt.plot(range(1, Ks+1), acc, 'g') 
plt.xlim(32, 47)  # Force x-axis to display 32 to 47
plt.xlabel('K Values')
plt.ylabel('Accuracy')
plt.xticks(range(32, 48)) #set x ticks to start from 32
plt.show() #the highest accuracy is for k=41, acc=0.385
#regardless of features we select, the model does not perform optimally.
#what is more, it performs better with a full set of features, wherefor k=38, acc=0.41 
