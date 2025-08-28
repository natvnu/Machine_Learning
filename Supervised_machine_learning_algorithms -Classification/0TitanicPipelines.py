
#Titanic passenger dataset to build a classification model 
#to predict whether a passenger survied the sinking of the Titanic
#1. install and import libraries

!pip install numpy
!pip install matplotlib
!pip install pandas
!pip install scikit-learn
!pip install seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#2. import dataset
titanic = sns.load_dataset('titanic')
titanic.head()

#deck has a lot of missing values so we'll drop it. 
#age has quite a few missing values as well. 
#embarked and embark_town don't seem relevant so we'll drop them
#It's unclear what alive refers to so we'll ignore it.
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
target = 'survived'
X = titanic[features]
y = titanic[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#3. develop pipelines
#Automatically detect numerical and categorical columns and assign them to separate numeric and categorical feature
numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),#uses median to replace the missing values. we use median because it works better than mean with dataset that have a lot of outliers
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
#combine the two into a single preprocessing ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
#combine a preprocessing and model into a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

#4.gridsearch
#define hyperparameters we want to explore
param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
}
# Cross-validation method
cv = StratifiedKFold(n_splits=5, shuffle=True)
best_model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print('***** Best Parameters *****')
print(best_model.best_params_) 
print('***** Classification Report *****')
print(classification_report(y_test, y_pred)) #{'classifier__max_depth': 10, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100}
print('***** Accuracy *****')
accuracy = best_model.score(X_test, y_test)
print(f"\nTest set accuracy: {accuracy:.2%}")

# Generate the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# Show the plot
plt.tight_layout()
plt.show()


#5. extract feature importances
#for this we need to navigate from GridSearch object back to estimator
# Step 1: Get the best fitted pipeline from the grid search
best_pipeline = best_model.best_estimator_
# Step 2: Extract the RandomForest classifier from the pipeline using its step name ('classifier')
best_random_forest = best_pipeline.named_steps['classifier']
# Step 3: Access the feature_importances_ attribute
feature_importances = best_random_forest.feature_importances_
feature_importances #an array of features

#to extract names of categorical features we need to go back to the one-hot-encoder 
#because that is where we transformed them
#continuous (numerical) features have not been transformed so we can access them directly
numerical_features
# Step 1: Get the best fitted pipeline from the grid search
best_pipeline = best_model.best_estimator_
# Step 2: Extract the preprocessor from the pipeline using its step name ('preprocessor')
best_preprocessor = best_pipeline.named_steps['preprocessor']
# Step 3: Access the catgorical transformer from the pipeline using its step name ('cat') 
categorical_preprocessor = best_preprocessor.named_transformers_['cat']
cat_features=categorical_preprocessor.named_steps['onehot'].get_feature_names_out(categorical_features)
#combine all feature names:
feature_names = numerical_features + list(cat_features)
feature_names

#combine features and feature names into a single dataframe
importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)
#plot the features and the importances 
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title('Most Important Features in predicting whether a passenger survived')
plt.xlabel('Importance Score')

plt.show()
