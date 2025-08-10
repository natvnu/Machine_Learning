
#build a model that predicts if a credit card transaction is fraudulent or not. 
#model the problem as a binary classification problem. 
#A transaction belongs to the positive class (1) if it is a fraud, otherwise it belongs to the negative class (0).
#The majority of the transactions are legitimate and only a small fraction are fradulent, 
#dataset that is highly unbalanced: only 0.172% of transactions are fraudulent
#284807 observations and 30 features in dataset + target var
'''
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
'''
# Import the libraries we need to use in this lab
#from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')

# download the dataset
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

# read the input data
raw_data=pd.read_csv(url)
raw_data.shape
'''
#display correlation to target var
corr_df=raw_data.corr()['Class'].drop('Class').sort_values(ascending=True)
corr_df.plot(kind='barh',figsize=(4, 6))
plt.show()# Amount,V13,V15,V22,V23,V25,V26 have the lowest correlation
# V17,V14,V12,V10,V16,V3 have the highest correlation
'''

#split the data
#drop the features with lowest correlation
#x=raw_data.drop(['Amount','V13','V15','V22','V23','V25','V26','Class'], axis=1)
#all features
#x=raw_data.drop(['Class'], axis=1)
#only the 6 most important features
x=raw_data[['V17','V14','V12','V10','V16','V3', 'Class']]
y=raw_data['Class']
#normalize x
x = normalize(x, norm="l1")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#calc sample weight to address imbalance in target var
w_train = compute_sample_weight('balanced', y_train)


#build dt classifier
# Initialize the classifier
dt = DecisionTreeClassifier(
    #criterion='gini',  # or 'entropy' for information gain
    max_depth=4,       # Prevents overfitting by limiting tree depth
    #min_samples_split=2,  # Minimum samples required to split a node
    random_state=42
)
# Train the model including sample weight to help with imbalance of target var
dt.fit(x_train, y_train, sample_weight=w_train) 
#make prediction
y_pred = dt.predict(x_test)
#evaluate the model
#we want to minimize false negatives -  recall
#we want a score that is robust to class imbalance (unlike accuracy)
auc = roc_auc_score(y_test, y_pred)
acu=accuracy_score(y_test, y_pred)
print('DT ROC-AUC Score:', auc)
# dropped lowest corr features: 0.7498593134496342, 0.9318070397827111 with sample weights
# all features: 0.7498681063590321, 0.9187958666865733 with sample weights
# 6 most important features: 1.0, 1.0 with sample weights => BEST OPTION FOR FEATURE SELECTION

#comparation with accuracy score, which should NOT be used in this case
#print('Accuracy Score:', acu) 
#dropped lowest corr features 0.9988588883817282, 0.9655384291281908 with sample weights

#build svm classifier
#scale features
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
# Initialize the classifier
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
# Train the model 
svm.fit(x_train, y_train)
#make prediction
y_pred_svm = svm.decision_function(x_test) #this can be used and the difference to the one below is HUGE
#y_pred_svm = svm.predict(x_test)
#evaluate svm
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print('SVM ROC-AUC score:', roc_auc_svm) 
# dropped lowest corr features 0.4999736212718064, 0.9755402794207159 with decision_function
# all features: 0.4985843415869443, 0.7885129431626337 with decision_function
# 6 most important features: 0.7920301069217783, 0.9933089548424885 with decision_function => BEST OPTION FOR FEATURE SELECTION

#Conclusion: use 6 most important features, normalize them, develop decision tree with max depth 4 and use sample weights for best results
