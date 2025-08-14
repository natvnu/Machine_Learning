
# Use numpy's random.seed() function to generate random data, make random clusters of points 
#by using the make_blobs class. Cluster the data


!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install plotly

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


np.random.seed(0)
x, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(x[:,0],x[:,1], marker='.',alpha=0.3, ec='k',s=80)
plt.show()

k_means = KMeans(init = "k-means++", #k-means++: Selects initial cluster centres for k-means clustering in a smart way to speed up convergence.
                 n_clusters = 4, #the number of clusters 
                 n_init = 12) #the number of times the k-means algorithm will be run with different centroid seed
#fit the model
k_means.fit(x)
#store labels (assigned clusters) in k_means_labels_
k_means_labels = k_means.labels_
k_means_labels
#store labels (assigned clusters) in k_means_cluster_centers
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

#plot the clusters
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))
# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the unique labels.
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))
# Create a plot
ax = fig.add_subplot(1, 1, 1)
# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(x[my_members, 0], x[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

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

