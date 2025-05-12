#Explore and prepare the dataset: Feature engineering and cleaning.
#Build a classifier pipeline: Model selection, training, and optimization.
#Evaluate the model’s performance: Interpret metrics and visualizations.
#Rainfall Prediction Classifier

#The original source of the data is Australian Government's Bureau of Meteorology and the latest data can be gathered from http://www.bom.gov.au/climate/dwo/.
#The dataset used in this project was downloaded from Kaggle at https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/
'''
#1.Import libraries
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install seaborn
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

#2.load dataset
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)

#3. clean the dataset - rows and values
df.shape #(145460, 23)
df.isnull().sum() #plenty of null values
#drop all rows with missing values
df=df.dropna()
df.isnull().sum() #no more null values
df.shape #left with (56420, 23)

#4. feature engineering 
#If we adjust our approach and aim to predict today’s rainfall instead of tomorrow's then we can make use of all features. (?!?)
#However, we need to update the names of the rain columns accordingly to avoid confusion.
df=df.rename(columns={'RainToday': 'RainYesterday', 'RainTomorrow': 'RainToday'})

#In the context of predicting whether it will rain today, we should consider removing features that are only fully known at the end of the day, 
#like MaxTemp, Rainfall, Evaporation, and Sunshine because they rely on the entire day's data. MinTemp, however, is typically measured in the morning 
#and could still be useful for predictions made earlier in the day. 
df=df.drop(['MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine'], axis=1)


#Because Australia is huge, weather patterns cannot have the same predictability in vastly different locations, 
#we will group together the 3 locations from below manually 
#can we use clustering here to group locations? - #YES, solution in RAIN_PREDICTION_Final_project_cities_clustering.ipynb
df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
#df. info()

###VERY CONVENIENT METHOD ######
#extract season from date to see how seasonality affects the rain prediction
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)

df=df.drop(columns=['Date'])
#df

###############################

#how balanced is our target?
df['RainToday'].value_counts() #No: 5766, Yes: 1791 - imbalanced

df_mel=df[df['Location']=='Melbourne']
df_mel.groupby(by='RainToday').count() #does not rain 1427 days, rains 471 days, not a balanced dataset, maybe use scaled weights, stratify y in train test split or use stratified cv

#5. model development - classifier
#define x and y
X=df.drop(['RainToday'],axis=1)
y=df['RainToday']
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#define preprocessing transformers for numerical and categorical features

#split numerical and categorical features in two lists
numerical_feat=X_train.select_dtypes(include='number').columns.tolist()
categorical_feat=X_train.select_dtypes(include='object').columns.tolist()

#define numerical and categorical preprocessing pipelines
numerical_pipeline=Pipeline(steps=[('scaler', StandardScaler())])
categorical_pipeline=Pipeline(steps=[('onehot',OneHotEncoder(handle_unknown='ignore'))])

#define preprocessing pipeline that contains numerical and categorical pipeline
Preprocessing_Pipeline = ColumnTransformer(
                        transformers=[
                        ('num', numerical_pipeline, numerical_feat),
                        ('cat', categorical_pipeline, categorical_feat)
                                      ])


#create a pipeline by combining the preprocessing with a Random Forest classifier

pipeline=Pipeline(steps=[
                    ('preprocessor', Preprocessing_Pipeline),
                    ('classifier',RandomForestClassifier(random_state=42))
                              ])

#define params for GridSearch
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

#define cv for GridSearch
cv=StratifiedKFold(n_splits=5,shuffle=True)

#perform GridSerach to find the best model
best_model = GridSearchCV(pipeline,cv=cv,param_grid=param_grid,scoring='accuracy',verbose=2)

#fit the best model
best_model.fit(X_train,y_train)

#predict
y_pred=best_model.predict(X_test)

#Print the best parameters and best crossvalidation score
print('Best parameters are: ', best_model.best_params_) #Best parameters are:  {'classifier__max_depth': None, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 50}
print('Best cross validation score is: ', best_model.best_score_) #Best cross validation score is:  0.8451612903225806
# Print estimated score of the best model
estimated_score = best_model.best_estimator_.score(X_test, y_test) #Estimated score is:  0.8346560846560847
print('Estimated score is: ', estimated_score)

#classification report
print('Best model classification report')
print(classification_report(y_test,y_pred))

#confusion matrix
print('Best model confusion matrix')
cf=confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cf)#this is also a good way to display confusion matrix, instead of heatmap
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show

#True positive rate is TPR=TP/TP+FN
#Extract TP, TN, FP, FN Values
tn, fp, fn, tp = cf.ravel()
#calc tpr
tpr = tp / (tp + fn)
print('True positive rate is:', tpr) #True positive rate is: 0.5

#6. Feature importances
#we have to work our way backward through the modelling pipeline to associate the feature importances with their one-hot encoded input features that were transformed from the original categorical features.
traced_back_categorical_feat=best_model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_feat) #5 features (categorical)
#we do not need to track numerical features because we did not change them, we only standardized their values
numerical_feat #4 features (numeric)
#caegorical and numerical features list
features=numerical_feat + list(traced_back_categorical_feat)
#WE CAN GET feature importances FROM THE MODEL  
#This also works for models developed without GridSearch, ie. rf.feature_importances_  where rf=RandomForestRegressor(n_estimators,random_state)
importances = best_model.best_estimator_['classifier'].feature_importances_ #9 importances
#create df with features and importances

df_importances=pd.DataFrame({'Feature': features,
                              'Importance': importances
                             }).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(16,20))
plt.bar(df_importances['Feature'],df_importances['Importance'])
plt.xlabel('Features')
plt.ylabel('Importances')
plt.xticks(rotation=90)
plt.title('Rain prediction in Australia - feature importances - RandomForestClassifier')
plt.show()


#9. Use LogisticRegression instead of RandomForestClassifier 

pipeline=Pipeline(steps=[
                    ('preprocessor', Preprocessing_Pipeline),
                    ('classifier',LogisticRegression(random_state=42))
                    ])

# Define a new grid with parameters for Logistic Regression parameters
param_grid = {
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}

#cv will stay the same
cv=StratifiedKFold(n_splits=5,shuffle=True)

#find best model through GridSearch
best_model = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)

#fit the best model
best_model.fit(X_train, y_train)

#make prediction
y_pred=best_model.predict(X_test)

#Print the best parameters and best crossvalidation score
print('Best parameters are: ', best_model.best_params_) #Best parameters are:  {'classifier__class_weight': None, 'classifier__penalty': 'l1', 'classifier__solver': 'liblinear'}

print('Best cross validation score is: ', best_model.best_score_) #Best cross validation score is:  0.835070306038048
# Print estimated score of the best model
estimated_score = best_model.best_estimator_.score(X_test, y_test)
print('Estimated score is: ', estimated_score) #0.8273809523809523

#classification report
print('Best model classification report')
print(classification_report(y_test,y_pred))


#confusion matrix
print('Best model confusion matrix')
cf=confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cf)#this is also a good way to display confusion matrix, instead of heatmap
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show

#True positive rate is TPR=TP/TP+FN
#Extract TP, TN, FP, FN Values
tn, fp, fn, tp = cf.ravel()
#calc tpr
tpr = tp / (tp + fn)
print('True positive rate is:', tpr) #True positive rate is: 0.5

##CONCLUSION: MODELS ARE VERY SIMILAR, SOME OF THE PARAMETERS ARE BETTER IN LOGISTIC REGRESSION, OTHERS ARE BETTER IN RANDOMFORESTCLASSIFIER
