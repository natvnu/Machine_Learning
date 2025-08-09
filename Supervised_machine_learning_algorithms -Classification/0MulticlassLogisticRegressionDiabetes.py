# 0MulticlassLogisticRegressionDiabetes
# dataset available at https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv

#Obesity Risk Prediction
#1. import libraries
'''
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install seaborn==0.13.2
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

#2. import dataset
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
data.head()

#3. preprocess data
#data.info()
#data.describe()
# HANDLING NUMERIC FEATURES - STANDARD SCALER
# extract numerical variables for scaling
numeric_data=data.select_dtypes(include=['float64','int64'])
#keep column names in numeric_data_cols
numeric_data_cols=numeric_data.columns
#scale numerical variables
scaler = StandardScaler()
scaler.fit(numeric_data)
scaled_array = scaler.transform(numeric_data)
#transform np array to df
scaled_df = pd.DataFrame(scaled_array, columns=numeric_data_cols)
#combine scaled columns with the original dataframe and remove the original columns
scaled_data = pd.concat([data.drop(columns=numeric_data), scaled_df], axis=1)
#now we have left categorical cols in their original state and scaled numerical cols
scaled_data
# HANDLING CATEGORICAL FEATURES - ONE-HOT ENCODING
#extract categorical variables
categorical_data=scaled_data.select_dtypes(include=['object'])
#remove target var
categorical_data=categorical_data.drop(['NObeyesdad'], axis=1)
#keep column names in categorical_data_columns
categorical_data_columns=categorical_data.columns
# Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first') #we use drop='first' to delete one of the columns
# This is done because if we are hot encoding Gender, encoded columns are Female and Male  
# If Female is 1, that means that Male must be 0, so we do not need to keep both columnsencoded_array = encoder.fit_transform(data[['Animal', 'Color']])
encoded_features = encoder.fit_transform(scaled_data[categorical_data_columns])
# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_data_columns))
# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_data_columns), encoded_df], axis=1)

# Encoding the target variable IMPORTANT!!!!
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
prepped_data.head()

#4. develop a model
#divide target var from features
y=prepped_data[['NObeyesdad']]
x=prepped_data.drop(['NObeyesdad'],axis=1)
#split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# train logistic regression model using One-vs-All (default)
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(x_train, y_train)
# Predictions
y_pred_ova = model_ova.predict(x_test)
# Evaluation metrics for OvA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")# Accuracy: 76.12%
#confusion matrix - OvA
class_order= model_ova.classes_ # the order of classes
cm = confusion_matrix(y_test, y_pred_ova, labels=class_order)
# heatmap for confusion matrix
sns.heatmap(cm, annot=True, cmap = 'RdBu', xticklabels=class_order, yticklabels=class_order)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# train logistic regression model using One-vs-One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(x_train, y_train)
# make predictions
y_pred_ovo = model_ovo.predict(x_test)
# evaluate
print("One-vs-All (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%") # Accuracy: 92.2%
#confusion matrix - OvO
class_order= model_ovo.classes_ # the order of classes
cm = confusion_matrix(y_test, y_pred_ovo, labels=class_order)
# heatmap for confusion matrix
sns.heatmap(cm, annot=True, cmap = 'RdBu', xticklabels=class_order, yticklabels=class_order)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

