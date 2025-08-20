
#Use the breast cancer data set included in scikit-learn to predict whether a tumor is benign or malignant
#Create two classification models and evaluate them.
#Interpreting and comparing the various evaluation metrics and the confusion matrix for each model will provide you with some valuable intuition regarding what the evaluation metrics mean and how they might impact your interpretation of the model performances.
'''
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install seaborn==0.13.2
'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = load_breast_cancer()
X, y = data.data, data.target
labels = data.target_names
feature_names = data.feature_names

'''
#sythetize df for easier preprocessing
df=pd.DataFrame(data.data, columns=data.feature_names)
df['class']=pd.DataFrame(data.target)
df.isnull().sum() #no null values
df.shape 
df.dtypes #all colums of correct datatypes

#features evaluation
#create df_corr df correlation to the target var only
df_corr=df_corr[['class']].abs().sort_values(by='class')
#plot the correlations
plt.figure(figsize=(15, 10))
df_corr.plot(kind='barh')
plt.show()
#most of the features seem to be well correlated with the target var, so we will try using them all
#however, removing some of the features (as below), SVM model returns even better results -> only 1 FN:
X=df.drop(['fractal dimension error', 'symmetry error', 'texture error', 'fractal dimension error','smoothness error', 'concavity error' ,'class'],axis=1)

'''

#develop classifiers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize the models
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', C=1, random_state=42)

# Fit the models to the training data
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

#predict
y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)

#evaluate
print(f"KNN Testing Accuracy: {accuracy_score(y_test, y_pred_knn):.3f}")
print(f"SVM Testing Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}")

print("\nKNN Testing Data Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nSVM Testing Data Classification Report:")
print(classification_report(y_test, y_pred_svm))

#plot confusion matrices
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)

#In medical testing and machine learning, we often frame the problem around detecting the "positive" class (the condition we're looking for). 
#In this case, instead, malignant is marked with 0 and benign with 1

axes[0].set_title('KNN Testing Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Testing Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

#conclusion
#we want to track maximize recall, which tracks a mass being malignant, of all the patients who actually have cancer, how many did our model correctly identify?
#It's the model's ability to find all the positive (malignant) cases. 
#The worse-case scenario then is a false negative prediction, where the test incorrectly predicts that the mass is benign.
#we want the TP to be as high as possible and FN to be as low as possible (0)
#in our case SVM returns 2 FN, which is better than 4 returned by KNN. 
#The fact that recall from clssification report shows higher value for SVM supports our conlusion
