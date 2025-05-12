'''

#Use scikit-learn's k-means clustering to cluster data
#Apply k-means clustering on a real world data for Customer segmentation


#1. import libraries
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install plotly
'''
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
import plotly.express as px

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

'''
#DATA GENERATION EXAMPLE
#2. generate data
np.random.seed(0)
#make random clusters of points by using the make_blobs class
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9) #X and y are numpy arrays
#X is two dimensional np array and y is one dimensional

#n_samples: The total number of points equally divided among clusters - 5000
#centres : The number of centres to generate, or the fixed centre locations - [[4, 4], [-2, -1], [2, -3],[1,1]]
#cluster_std: The standard deviation of the clusters - 0.9

#plot the X data (X[:,0] is first feature, X[:,1] is second feature) just to se ehow our data look like
#plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.3,ec='k',s=80)

#3. develop the K-means clustering model
#create Kmeans model
k_means=KMeans(init='k-means++',n_clusters=5, n_init=12)
k_means.fit(X)
k_means_labels=k_means.labels_ #np array of labels, or class of every single data point (0,1,2,3) in our case, or clusters to which the data point belongs
k_means_cluster_centres=k_means.cluster_centers_ #np array of coordinates of cluster centres

#4. plotting the model
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))
# Create a plot
ax = fig.add_subplot(1, 1, 1)

#loop through centeroids and colors:
for k, col in zip(range(len(k_means.cluster_centers_)), colors): #range(len(k_means.cluster_centers_)) returns range(0,4) when we have 4 cluster centroids, (0,3) when we have 3 centroids, etc
    my_members=(k_means_labels==k) #np array of True, False. True for members that belong to class k

    # Plots the datapoints with colors from colors np array
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

    #define cluster_centre
    cluster_center = k_means_cluster_centres[k]

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

# Create a plot
ax = fig.add_subplot(1, 1, 1)

#loop through centeroids and colors:
for k, col in zip(range(len(k_means.cluster_centers_)), colors): #range(len(k_means.cluster_centers_)) returns range(0,4) when we have 4 cluster centroids, (0,3) when we have 3 centroids, etc
    my_members=(k_means_labels==k) #np array of True, False. True for members that belong to class k

    # Plots the datapoints with colors from colors numpy array 
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

    #define cluster_centre
    cluster_center = k_means_cluster_centres[k]

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()
'''
#DATA IMPORT EXAMPLE
#applying customer segmentation to historical data
#2. load and preprocess the dataset
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")
cust_df.dtypes #address is an object, what is more it is categorical var. We could hot encode it like this:
'''
#hot encoding the Address column
from sklearn.preprocessing import OneHotEncoder
# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Fit and transform the data
one_hot_encoded = encoder.fit_transform(cust_df[["Address"]])

# Create a DataFrame with the encoded data
df_encoded = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(["Address"]))
#concatenate encoded dataframe with original one
cust_df_pp = pd.concat([cust_df, df_encoded], axis=1)
'''

#but in this case we will just drop it
#dropping the Address column
cust_df = cust_df.drop('Address', axis=1)
# Drop NaNs from the dataframe
cust_df = cust_df.dropna() # drops rows with nan (null) values, regardless of in which column the nan values are
#cust_df.info() #no more nan values
#define x
cust_df=cust_df.drop('Customer Id',axis=1) #Customer Id makes no impact to the model prediction
from sklearn.preprocessing import StandardScaler
# Initialize the StandardScaler
scaler = StandardScaler()
# Fit the scaler
x=scaler.fit_transform(cust_df)

#3. develop the K-means clustering model
k_means=KMeans(init='k-means++',n_clusters=3,n_init=12)
k_means.fit(x)
k_means_labels=k_means.labels_
k_means_labels
cust_df['Cluster'] = k_means_labels
cust_df
#we can group all rows by cluster and find average values per each column
cust_df.groupby('Cluster').mean()
'''
#Now, let's look at the distribution of customers based on their education, age and income. 
#We can choose to visualise this as a 2D scatter plot with Age on the x-axis, 
#Income on the y-axis and the marker size representing education. 
#The scatter points will be assigned different colors based on different class labels
plt.scatter(x[:,0], x[:,3], #x is np array, we cannot use x['Age'] and x['Income']
            s=x[:,1]**3,
            c=k_means_labels,
             alpha=0.5)
plt.xlabel("Age", size=16)
plt.ylabel("Income", size=16)
plt.title("Bubble Plot Age vs Income, size is Education and Color is Cluster", size=18)
plt.show()
'''
# Create interactive 3D scatter plot (?!?)
fig = px.scatter_3d(x, x=1, y=0, z=3, opacity=0.7, color=k_means_labels.astype(float))

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Education'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ))  # Remove color bar, resize plot

fig.show()


#Create a profile for each cluster, for example:
#LATE CAREER, AFFLUENT, AND EDUCATED
#MID CAREER AND MIDDLE INCOME
#EARLY CAREER AND LOW INCOME
#cust_df.groupby('Cluster').mean() actually provides us with more useful info than both 2D and 3D graph
