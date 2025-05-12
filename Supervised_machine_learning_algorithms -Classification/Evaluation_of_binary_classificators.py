
'''
In this project we will:
Use the breast cancer data set included in scikit-learn to predict whether a tumor is benign or malignant
Create two classification models and evaluate them.
Add some Gaussian random noise to the features to simulate measurement errors
Interpret and compare the various evaluation metrics and the confusion matrix for each model to understand intuition regarding what the evaluation metrics mean and how they might impact  interpretation of the model performances.
The goal in this lab is not to find the best classifier - it is primarily intended for you to practice interpreting and comparing results in the context of a real-world problem
'''

#1. install and import libraries
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install seaborn==0.13.2

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer # used to load Breast Cancer Wisconsin dataset, a straightforward binary classification dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#2. load dataset
data=load_breast_cancer() #data is of Bunch datatype
X, y = data.data, data.target #X and y are of np array datatype
labels = data.target_names #2 classes, malignant or benign
feature_names = data.feature_names
X.shape #30 features and 569 observations
y.shape #1 target var and 569 observations
#print(data.DESCR) #prints detailed data description, explanation of each feature
'''
#we can do the below to display X array in a form of a dataframe, but it is not necessary
df_X=pd.DataFrame(X)
df_X.columns=[feature_names]
df_X.head()
df_y=pd.DataFrame(y)
df_y.columns=['Class']
df_y.head()
'''

#3.prep the data and add some noise
#standardize X
X_scaled=StandardScaler().fit_transform(X)

#add some noise
# Add Gaussian noise to the data set
np.random.seed(42)  # For reproducibility
noise_factor = 0.5 # Adjust this to control the amount of noise
X_noisy = X_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

# Load the original and noisy data sets into a DataFrame for comparison and visualization
df = pd.DataFrame(X_scaled, columns=feature_names)
df_noisy = pd.DataFrame(X_noisy, columns=feature_names)

#print both dataframes for comparison
df.head()
df_noisy.head()

#4. compare the original data (df) with data with noise (df_noisy) #not necessary
'''
#compare two dfs using histogram
plt.figure(figsize=(12, 6))
# Original Feature Distribution (Noise-Free)
plt.subplot(1, 2, 1)
plt.hist(df[feature_names[5]], bins=20, alpha=0.7, color='blue', label='Original') #feature_names[5]=>'mean compactness', so we are using only 1 column to compare
plt.title('Original Feature Distribution')
plt.xlabel(feature_names[5])
plt.ylabel('Frequency')
# Noisy Feature Distribution
plt.subplot(1, 2, 2)
plt.hist(df_noisy[feature_names[5]], bins=20, alpha=0.7, color='red', label='Noisy') 
plt.title('Noisy Feature Distribution')
plt.xlabel(feature_names[5])  
plt.ylabel('Frequency')
plt.tight_layout()  # Ensures proper spacing between subplots
plt.show()

#compare two dfs using line chart
plt.figure(figsize=(12, 6))
plt.plot(df[feature_names[5]], label='Original',lw=3)#again, wea re using only feature_names[5], aka 'mean compactness' column
plt.plot(df_noisy[feature_names[5]], '--',label='Noisy',)
plt.title('Scaled feature comparison with and without noise')
plt.xlabel(feature_names[5])
plt.legend()
plt.tight_layout()
plt.show()


#check the correlation between two dfs using scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(df[feature_names[5]], df_noisy[feature_names[5]],lw=5)
plt.title('Correlation between scaled feature with and without noise')
plt.xlabel('Original Feature')
plt.ylabel('Noisy Feature')
plt.tight_layout()
plt.show()

'''
# 5. split the data and develop 2 classifiers: KNN and SVM models using noisy training data
#splitting the data
x_train,x_test, y_train, y_test=train_test_split(X_noisy,y,test_size=0.3,random_state=42)

#knn model fitting and prediction 
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
yhat_knn=knn.predict(x_test) #prediction on test data

#svm model fitting and prediction
svc=SVC(kernel='linear', C=1, random_state=42)
svc.fit(x_train,y_train)
yhat_svm=svc.predict(x_test) #prediction on test data

#6. Evaluate the classifiers on test data
print(f"KNN Testing Accuracy: {accuracy_score(y_test, yhat_knn):.3f}") #0.936
print(f"SVM Testing Accuracy: {accuracy_score(y_test, yhat_svm):.3f}") #0.971 SVM MODEL IS BETTER

print("\nKNN Testing Data Classification Report:")
print(classification_report(y_test, yhat_knn))

print("\nSVM Testing Data Classification Report:")
print(classification_report(y_test, yhat_svm))#SVM MODEL IS BETTER

#print("\nKNN Testing Data Confusion Matrix:")
#print(confusion_matrix(y_test,yhat_knn))

#print("\nSVM Testing Data Confusion Matrix:")
#print(confusion_matrix(y_test,yhat_svm))#SVM MODEL IS BETTER

#plot the confusion matrices for knn and svm - evaluation based on test data
fig, ax = plt.subplots(1, 2)
sns.heatmap(confusion_matrix(y_test,yhat_knn), annot=True, cmap='Blues', fmt='d', ax=ax[0],
            xticklabels=labels, yticklabels=labels)
ax[0].set_title('KNN Testing Confusion Matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test,yhat_svm), annot=True, cmap='Blues', fmt='d', ax=ax[1],
            xticklabels=labels, yticklabels=labels)

ax[1].set_title('SVM Testing Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.tight_layout() #without this some elements overlap
plt.show()

#the worst kind of error that can come out of this classification is to classify a cell as benign when it is actually malignant - false negative quadrant.
#in knn classifier the number of these errors are 7, while in svm classifiers the number of this error is 2
#svm has lower false negative result, which is extremely imporant in this case, but also has higher accuracy, precision, recall, F1-score


#7. check if we are overfitting the model by comparing evaluation on test data (above) and train data (below)
#make the prediction using the training data again
yhat_knn_tr=knn.predict(x_train)
yhat_svc_tr=svc.predict(x_train)

#evaluate models on training data
#accuracy
print(f"KNN Training Accuracy: {accuracy_score(y_train, yhat_knn_tr):.3f}") #0.955
print(f"SVM Training Accuracy: {accuracy_score(y_train, yhat_svc_tr):.3f}") #0.972 SVM MODEL IS BETTER

#classification report
print('\nKNN Training Data Classification Report')
print(classification_report(yhat_knn_tr,y_train))
print('\nSVM Training Data Classification Report')
print(classification_report(yhat_svc_tr,y_train))



#plot the confusion matrices for knn and svm - evaluation based on train data
fig,ax=plt.subplots(1,2)
sns.heatmap(confusion_matrix(y_train,yhat_knn_tr), cmap='Blues', annot=True, #without annotations TP,TN, FP and FN numbers do not show on the graph
            fmt='d', #without fmt numbers are in the form 1e+02
            xticklabels=labels, yticklabels=labels, ax=ax[0]) 
ax[0].set_title('KNN Training Confusion Matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_train,yhat_svc_tr), cmap='Blues', annot=True, #without annotations TP,TN, FP and FN numbers do not show on the graph
            fmt='d', #without fmt numbers are in the form 1e+02
            xticklabels=labels, yticklabels=labels, ax=ax[1]) 

ax[1].set_title('SVM Training Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()

'''
Ideally the accuracy of a model would be almost the same on the training and testing data sets.

It would be unusual for the accuracy to be higher on the test set and this might occur due to chance or some sort of data leakage. 
For example, here we have normalized all of the data, instead of fitting StandardScaler to the training data and only then applying it to the train and test sets separately. 
We'll revisit this and other pitfalls in another project.

When the accuracy is substantially higher on the training data than on the testing data, 
the model is likely memorizing details in the training data that don't generalize to the unseen data - the model is overfitting to the training data.

Model	Phase	Accuracy
KNN	Train	95.5%
KNN	Test	93.6%
SVM	Train	97.2%
SVM	Test	97.1%
For the SVM model, the training and testing accuracies are essentially the same at about 97%. This is ideal - the SVM model is likely not overfit. 
For the KNN model, however, the training accuracy is about 2% higher that the test accuracy, indicating there might be some overfitting.

In summary, the SVM model is both more convincing and has a higher accuracy than the KNN model. 
We aren't trying to tune these models; we are just comparing their performance with a fixed set of hyperparamters.
'''
