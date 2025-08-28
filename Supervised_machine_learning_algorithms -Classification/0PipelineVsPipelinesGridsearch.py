# Generate synthetic data for running targeted experiments using scikit-learn
#Train and evaluate a KNN classification model using a pipeline
#Tune model hyperparameters using a pipeline within a cross-validation grid search
#Build a more complex random forest classification pipeline using real-world data
#Extract the feature importances from the trained pipeline

#Pipeline class from Scikit-Learn is invaluable for streamlining data preprocessing 
#and model training into a single, coherent sequence. 
#A pipeline is essentially a sequence of data transformers that culminates with an optional final predictor. 
#A major advantage of using a pipeline is that it enables comprehensive cross-validation 
#and hyperparameter tuning for all steps simultaneously. By integrating the pipeline 
#within GridSearchCV, you can fine-tune not only the model but also the preprocessing 
#steps, leading to optimized overall performance. 

#1. install and import libraries
'''
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install seaborn==0.13.2
'''
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix

#2. import iris dataset
data = load_iris()
X, y = data.data, data.target
labels = data.target_names

#3. develop a model
#Instantiate a pipeline consisting of StandardScaler, PCA, and KNeighborsClassifier
pipeline = Pipeline([
        ('scaler', StandardScaler()),       # Step 1: Standardize features
        ('pca', PCA(n_components=2),),       # Step 2: Reduce dimensions to 2 using PCA
        ('knn', KNeighborsClassifier(n_neighbors=5,))  # Step 3: K-Nearest Neighbors classifier
    ])

#split the data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#fit pipeline on training set
pipeline.fit(X_train, y_train)
#predict
y_pred = pipeline.predict(X_test)

# Measure the pipeline accuracy on the test data
test_score = pipeline.score(X_test, y_test)
print(f"{test_score:.3f}") # accuracy: 0.9 which is good

# generate the confusion matrix and plot it
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.title('KNN Pipeline Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show() #10 are accurately classified as setosa, 9 as versicolor and 8 as virginica, 
           #2 are inaccurately classified to be versicolor but are virginica
           #1 is inaccurately classified as virginica but is versicolor

#4. find the best parameters (refine the model) using using grid search
# make a pipeline without specifying any parameters 
pipeline = Pipeline(
                    [('scaler', StandardScaler()),
                     ('pca', PCA()),
                     ('knn', KNeighborsClassifier()) 
                    ]
                   )

#define parameters grid
param_grid = {'pca__n_components': [2, 3],
              'knn__n_neighbors': [3, 5, 7]
             }

#in case target var is imbalanced, we can use StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# initialize GridSearch 
best_model = GridSearchCV(estimator=pipeline,
                          param_grid=param_grid,
                          cv=cv,
                          scoring='accuracy',
                          verbose=2
                         )
#fit grid search
best_model.fit(X_train, y_train)
#predict
y_pred_gs=best_model.predict(X_test)
# Measure the pipeline accuracy on the test data
test_score = best_model.score(X_test, y_test)
print(f"{test_score:.3f}") # accuracy: 0.933 which is better
#print best parameters
print('Best parameters are: ', best_model.best_params_) #{'knn__n_neighbors': 3, 'pca__n_components': 3}

#confusion matrix
conf_matrix=confusion_matrix(y_test, y_pred_gs)
# Create a single plot for the confusion matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.title('KNN Pipeline Confusion Matrix with GridSearch')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show() #10 are accurately classified as setosa, 10 as versicolor and 8 as virginica, 
           #2 are inaccurately classified to be versicolor but are virginica

#model refined by gridsearch is performing better
