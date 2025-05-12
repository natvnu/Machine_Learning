#during their course of treatment, each patient responded to one of 5 medications: Drug A, Drug B, Drug C, Drug X and Drug Y.
#build a  multiclass classifier to find out which drug might be appropriate for a future patient with the same illness
#The features are the Age, Sex, Blood Pressure, and Cholesterol of the patients, and the target is the drug that each patient responded to


#1. import libraries
!pip install numpy
!pip install pandas 
!pip install matplotlib
!pip install scikit-learn
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

#2. import dataset
path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)
my_data

#3 data wrangling
#my_data.info() #4 of the features are categorical, cannot be used in skicit learn
#transform categorical data
'''
#handle indicator values with get_dummies - 1st way
#for every column creates as many new cols as there were unique values and these new cols are boolean 
#ie [sex (female,male)] => [sex_female (True,False)] and [sex_male (True,False)]
get_dummies_cd=['Sex','BP','Cholesterol','Drug']
my_data_new=pd.get_dummies(my_data,get_dummies_cd)
my_data_new.info()
'''
#handle indicator values with LabelEncoder - 2nd way I LIKE IT BETTER
#does not create new cols, just changes the values from object to int
#ie [Cholesterol (high,medium,low)] =>[Cholesterol (2,1,0)]
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex']) 
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol']) 
#target var we have not changed, but have instead added one numerical column with int values corresponding to object values in original column
my_data['Drug_num'] = label_encoder.fit_transform(my_data['Drug']) 

#4. EDA
#explore the correlation between features and the target var
my_data[['Age','Sex','BP','Cholesterol','Na_to_K','Drug_num']].corr() #Na_to_K and BP have the strongest corr with target var

#5. Building a decision tree model
#define x and y
x=my_data.drop(['Drug','Drug_num'], axis=1)
y=my_data['Drug']
#train-test split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3, random_state=32)
dt = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
dt.fit(x_train, y_train)
yhat_dt=dt.predict(x_test)
#calculate the score of the dt
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_test, yhat_dt)) #0.9833333333333333
#print the tree
plot_tree(dt) #x[4] is 5th column, Na_to_K, value [0,0,0,0,63] is 63 samples of Drug Y (Drug A would be in the first place, ie (3,0,0,0,0))
plt.show() # so Drug Y is determined by Na_to_K more than 14.627 
# the decision criterion for Drug Y is Na_to_K>14.627. Looking at the dt we can figure out the rest as well
