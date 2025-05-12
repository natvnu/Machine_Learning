#use the Titanic Survival Dataset to build a classification model to predict whether a passenger 
#survived the sinking of the Titanic, based on attributes of each passenger in the data set.
#start with building a Random Forest Classifier, then modify your pipeline 
#to use a Logistic Regression estimator instead. 
#You'll evaluate and compare your results.
#use cross validation and a hyperparameter grid search to optimize your machine learning pipeline

#1. import libraries
#2. import dataset
#3. preprocess data (null values, are the vars balanced, etc...)
#4. train test split (stratify y if needed)
#5. create data pretprocessing pipelines for numeric and categorical values
#6. create pipeline that includes Preprocessing and RandomForest Classifier, set parameters and choose CrossValidation method - CV
#use pipeline, params and CV as input to GridSearch to directly find hyperpars, train it and then evaluate results
#7. find feature importances
#8. create pipeline that includes Preprocessing and Logistic Regression Classifier, set parameters and choose CrossValidation method - CV
#use pipeline, params and CV as input to GridSearch to directly find hyperpars, train it and then evaluate results


#1.import libraries
'''
!pip install numpy
!pip install matplotlib
!pip install pandas
!pip install scikit-learn
!pip install seaborn
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


#2. import dataset
titanic = sns.load_dataset('titanic')

#3. preprocess data (null values, are the vars balanced, etc...)
titanic.isnull().sum() 
#age has 177 missing values, deck has 688 (out of 891 observations),
#embarked and embark_town don't seem relevant, alive is not clear what it refers to, so we will drop all these
titanic=titanic.drop(['deck','age','embark_town','embarked','alive'],axis=1)
titanic.isnull().sum()#no null values anymore
'''
# we can use this instead of 'onehot',OneHotEncoder(handle_unknown='ignore' in step 5
#sex, class and who are non-numerical categorical values, so let's deal with them
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
titanic['sex']=labelencoder.fit_transform(titanic['sex'])
titanic['who']=labelencoder.fit_transform(titanic['who'])
titanic['class']=labelencoder.fit_transform(titanic['class'])
titanic #now these columns are still categorical, but they are now numerical (integers, not objects)
#

## we can use this for assignement of target and features and instead of ('scaler',StandardScaler()) in step 5
#assign target and features
x_raw=titanic.drop(['survived'],axis=1)
y=titanic['survived']
#scale the features
x=StandardScaler().fit_transform(x_raw)
##
-------
### if we do use it then we can DELETE step 5 and in step 6 we can change pipeline like this:
pipeline=Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('classifier',RandomForestClassifier(random_state=42))      
                        ])
#define parameters and everything after that will be the same
###


#### RESULTS WILL NOT BE THE SAME, we will have
#23 predicted as not survived, but they actually did, 21 predicted as survived, but they actually did not
#### BUT THAT IS OK, especially if we would rather avoid so many pipelines



##### in step 7, features collection for calculating feature importances will be different.
In our case we did not change features (columns), we only changed values (from object to int). 
So our columns have not changed compared to original features dataframe x_raw. 
Instead of  traced_back_categorical_feat .... we  will use:

feature_importances = best_model.best_estimator_['classifier'].feature_importances_ #9 importances
feature_names=x_raw.columns.tolist()

#importance_df = pd.DataFrame... and plot will remain exactly the same
#####
'''


#assign target and features
x=titanic.drop(['survived'],axis=1)
y=titanic['survived']

#how balanced is the target
y.value_counts() #around 38% of passengers survived, it makes data slightly imbalanced, therefore we should stratify it when performing train test split

#4.train test split the data
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#5. create data pretprocessing pipelines for numeric and categorical values
#split numerical and categorical features in two lists
numerical_feat=x_train.select_dtypes(include='number').columns.tolist()#choose numerical features and transform this set into a list
categorical_feat=x_train.select_dtypes(include=['object']).columns.tolist()#categorical features
#does select_dtypes(include=['object']) atually includes all categorical features or only those that are objects?


#define pipelines separately for each of the lists of features
numerical_pipeline=Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler',StandardScaler())
                             ])

categorical_pipeline=Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot',OneHotEncoder(handle_unknown='ignore'))     
                             ])
                             
#define preprocessing pipeline that consists of the numerical and categorical pipelines
preprocessing_pipeline=ColumnTransformer(transformers=[
                            ('num', numerical_pipeline, numerical_feat),
                            ('cat',categorical_pipeline, categorical_feat)
                            ])
    

#6. create pipeline that includes Preprocessing and RandomForest Classifier, set parameters and choose CrossValidation method - CV
#use pipeline, params and CV as input to GridSearch to directly find hyperpars, train it and then evaluate results
pipeline=Pipeline(steps=[
                            ('preprocessor', preprocessing_pipeline),
                            ('classifier',RandomForestClassifier(random_state=42)) #PAY ATTENTION NOT TO USE RandomForestRegressor
                        ])
    #define parameters                          
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

#define method for cross validation
cv=StratifiedKFold(n_splits=5, shuffle=True)


#define GridSearch estimator
best_model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)

#fit GridSearch estimator
best_model.fit(x_train, y_train)

#predictions from the GridSearch estimator on test data
y_pred=best_model.predict(x_test)

#print out classification report
print(classification_report(y_test,y_pred))

#Plot the confusion matrix
conf_matrix=confusion_matrix(y_test,y_pred)
conf_matrix

plt.figure()
sns.heatmap(conf_matrix,annot=True,cmap='Blues',fmt='d')
plt.title('Pipeline classification confusion matrix for survivers on Titanic - Random Forest Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()#21 predicted as not survived, but they actually did,16 predicted as survived, but they actually did not


#7. Feature importances
#we have to work our way backward through the modelling pipeline to associate the feature importances with their one-hot encoded input features that were transformed from the original categorical features.
traced_back_categorical_feat=best_model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_feat) #5 features (categorical)
#we do not need to track numerical features because we did not change them, we only standardized their values
numerical_feat #4 features (numeric)

#WE CAN GET feature importances FROM THE MODEL. 
#This also works for models developed without GridSearch, ie. rf.feature_importances_  where rf=RandomForestRegressor(n_estimators,random_state)
feature_importances = best_model.best_estimator_['classifier'].feature_importances_ #9 importances

# Combine the numerical and traced back categorical feature names
feature_names = numerical_feat + list(traced_back_categorical_feat) #all 9 features are now here

#lets plot the features and their importances
#create df with feature importances
importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)
#plot
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title('Most Important Features in predicting whether a passenger survived - Random Forest Classifier')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.show()

# Print test score
test_score = best_model.best_estimator_.score(x_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}") # 77.65%

#8. change the estimator in pipeline so that includes Preprocessing and Logistic Regression, set parameters and choose CrossValidation method - CV
#use pipeline, params and CV as input to GridSearch to directly find hyperpars, train it and then evaluate results
pipeline=Pipeline(steps=[
                    ('preprocessor', preprocessing_pipeline),
                    ('classifier',LogisticRegression(random_state=42))    
                   ])
 
param_grid={ 
            'classifier__solver' : ['liblinear'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__class_weight' : [None, 'balanced']
            }

#define method for cross validation
cv=StratifiedKFold(n_splits=5, shuffle=True)

#define GridSearch estimator
best_model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)


#fit GridSearch estimator
best_model.fit(x_train, y_train)

#predictions from the GridSearch estimator on test data
y_pred=best_model.predict(x_test)

#print out classification report
print(classification_report(y_test,y_pred))

#Plot the confusion matrix
conf_matrix=confusion_matrix(y_test,y_pred)
conf_matrix

plt.figure()
sns.heatmap(conf_matrix,annot=True,cmap='Blues',fmt='d')
plt.title('Pipeline classification confusion matrix for survivers on Titanic - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()#18 predicted as not survived, but they actually did,11 predicted as survived, but they actually did not

#9. Extract the logistic regression feature coefficients and plot their magnitude in a bar chart.
#categorical features traced back
traced_back_categorical_feat=best_model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_feat) #5 features (categorical)
# Combine the numerical and traced back categorical feature names
feature_names = numerical_feat + list(traced_back_categorical_feat) #all 9 features are now here
#extract feature importances PAY ATTENTION THAT THEY ARE CALLED as coef[0] in case of LOG REGRESSION
coefficients = best_model.best_estimator_.named_steps['classifier'].coef_[0]

#create df with feature importances
importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': coefficients
                             }).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.bar(importance_df['Feature'],importance_df['Importance'],color='skyblue')
plt.title('Most Important Features in predicting whether a passenger survived - Logistic Regression')
plt.xlabel('Features')
plt.ylabel('Importances')
plt.tight_layout()
plt.show()

# Print test score
test_score = best_model.best_estimator_.score(x_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}") #83.8%

###CONCLUSION coef matrix of RandomForest Classifier has more errors than Logistic Regression, also it's accuracy is 77.65%, while Logistic Regression accuracy is 83.8%
