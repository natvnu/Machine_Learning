'''
Generate synthetic data for running targeted experiments using scikit-learn
Create k-means models and evaluate their comparative performance
Investigate evaluation metrics and techniques for assessing clustering results
'''

#1. Import libraries
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install scipy==1.14.1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Patch
from matplotlib import cm


#2. Define function for evaluating the clustering models we'll be building. 
#We'll include silhouette scores and the Davies-Bouldin index, 
#plus generate a plot displaying the silhouette scores
def evaluate_clustering(X, labels, n_clusters, ax=None, title_suffix=''):
    """
    Evaluate a clustering model using silhouette scores and the Davies-Bouldin index.
    
    Parameters:
    X (ndarray): Feature matrix.
    labels (array-like): Cluster labels assigned to each sample.
    n_clusters (int): The number of clusters in the model.
    ax: The subplot axes to plot on.
    title_suffix (str): Optional suffix for plot titlec
    
    Returns:
    None: Displays silhoutte scores and a silhouette plot.
    """
    if ax is None:
        ax = plt.gca()  # Get the current axis if none is provided
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    # Plot silhouette analysis on the provided axis
    unique_labels = np.unique(labels)
    colormap = cm.tab10
    color_dict = {label: colormap(float(label) / n_clusters) for label in unique_labels}
    y_lower = 10
    for i in unique_labels:
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = color_dict[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_title(f'Silhouette Score for {title_suffix} \n' + 
                 f'Average Silhouette: {silhouette_avg:.2f}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlim([-0.25, 1])  # Set the x-axis range to [0, 1]

    ax.set_yticks([])

#3. Create synthetic data with four blobs to experiment with k-means clustering
#blobs will be slightly overlapping
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=[1.0, 3, 5, 2], random_state=42)


#4. Develop k-means model and divide data points into clusters
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
kmeans_labels=kmeans.labels_
colormap = cm.tab10

#5. display original data as blobs, data assigned to clusters and evaluation of clustering using the silhouette plot 
# Plot the blobs
# fig1 is blobs non clustered: X[:,0] is feature 1 - on x axis and X[:,1] is feature 2 on y axis
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.6, edgecolor='k')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', alpha=0.9, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

#fig2 is blobs clustered in 4 clusters
plt.subplot(1, 3, 2)
#plot clusters
plt.scatter(X[:, 0], X[:, 1], s=50, 
            c=kmeans_labels,
            alpha=0.6, edgecolor='k')
#plot cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='white', s=200, marker='o', alpha=0.9, edgecolor='k', label='Centroids')
# Label the custer number
for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
plt.title(f'Synthetic Blobs with {n_clusters} Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

#fig3 is evaluation of the clustering
plt.subplot(1, 3, 3)
evaluate_clustering(X, y_kmeans, n_clusters, title_suffix=' k-Means Clustering')
plt.show()

#Each point in a silhouette plot has a silhouette score ranging from -1 to 1. 
#A high silhouette score indicates the data point is much more similar to its own cluster than its neighboring clusters. 
#A score near 0 implies the point is at or near the decision boundary between two clusters. A negative score means the point might have been assigned to the wrong cluster. 
#We'll take a closer look at the silhoutte plot later.


#6. determine cluster stability - how do the results change when K-means is run using different initial centroid seeds
# to evaluate the stability of clustering, running k-means multiple times with different initial centroids by not fixing the random state
#it helps determine if the algorithm consistently produces similar cluster assignments and inertia scores
#consistent inertia across runs suggests a stable solution that is less dependent on initial centroid positions

# Number of runs for k-means with different random states
n_runs = 8
inertia_values = []

# Calculate number of rows and columns needed for subplots
n_cols = 2 # Number of columns
n_rows = 4#-(-n_runs // n_cols) # Ceil division to determine rows

plt.figure(figsize=(18, 10)) # Adjust the figure size for better visualization

#we will run K-Means multiple times with different random states and keep inertia scores in the array, in addition, for each run we will plot the cluster result
for i in range(n_runs):
    kmeans = KMeans(n_clusters=4, random_state=None) # Use the default `n_init`
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    

    
    # Plot the clustering result 
    plt.subplot(2,4,i+1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='tab10', alpha=0.6, edgecolor='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='x', label='Centroids')
    plt.title(f'K-Means Run {i + 1}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
plt.show()

# Print inertia values
for i, inertia in zip(range(n_runs),inertia_values):
    print('Run: ',(i+1), 'Inertia: ', inertia)
#as we can see on the subplots, the cluster assignments vary between runs when using different initial centroid seeds. 
#Inertia values also show inconsistency, indicating that the clustering process is sensitive to the initial placement of centroids. 
#This all means the result is not too reliable


#7. let's see how inertia value, silhouette score and Davies Bouldin Index changes with the number of clusters

#create arrays to store inertia value, silhouette score and Davies Bouldin Index results for every iteration (for every no of clusters)
inertia_values=[]
silhouette_values=[]
dbi_values=[]
k_values = range(2, 11)

for n in k_values:
    kmeans=KMeans(n_clusters=n, random_state=None) 
    kmeans.fit(X)
    y_kmeans=kmeans.fit_predict(X)
    inertia_values.append(kmeans.inertia_)
    silhouette_values.append(silhouette_score(X, y_kmeans))
    dbi_values.append(davies_bouldin_score(X, y_kmeans))
#plot line diagram of inertia_values vs number of clusters
plt.figure(figsize=(10, 4))
plt.subplot(1,3,1)
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method:\n Inertia vs. number of clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

plt.subplot(1,3,2)
plt.plot(k_values, silhouette_values, marker='o')
plt.title('Silhouette score vs. number of clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette score')

plt.subplot(1,3,3)
plt.plot(k_values, dbi_values, marker='o')
plt.title('DBI vs. number of clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davides Bouldin Index')

plt.tight_layout()
plt.show()
#conclusion: Inertia we want as low as possible. In out case it stars to settle around k=3 or 4 (it is still dropping, but not significantly)
#Silhouette score we want high. So, k=3 gives the best result
#Davies Bouldin index we want as low as possible, so again, k=3 gives us the optimal model


#8. Plot the blobs and the clustering results for k = 3, 4, and 5

#set the figuresize
plt.figure(figsize=(10,10))

#k=3
kmeans=KMeans(n_clusters=3, random_state=42)
kmeans.fit_transform(X)
y_means=kmeans.predict(X)
centroids=kmeans.cluster_centers_
colors=kmeans.labels_
   
#plot the blobs
plt.subplot(3,2,1)
plt.scatter(X[:,0],X[:,1],c='darkblue', edgecolor='k')
plt.scatter(centroids[:,0], centroids[:,1],c='r', marker='o')
plt.title('Unclustered data points')


#plot the clusters (k=3)
plt.subplot(3,2,2)
plt.scatter(X[:,0],X[:,1],c=colors, edgecolor='k')
plt.scatter(centroids[:,0], centroids[:,1],c='r', marker='o')
plt.title('Data appointed to 3 clusters')

#k=4
kmeans=KMeans(n_clusters=4, random_state=42)
kmeans.fit_transform(X)
y_means=kmeans.predict(X)
centroids=kmeans.cluster_centers_
colors=kmeans.labels_

#plot the blobs
plt.subplot(3,2,3)
plt.scatter(X[:,0],X[:,1],c='darkblue', edgecolor='k')
plt.scatter(centroids[:,0], centroids[:,1],c='r', marker='o')
plt.title('Unclustered data points')

#plot the clusters (k=4)
plt.subplot(3,2,4)
plt.scatter(X[:,0],X[:,1],c=colors, edgecolor='k')
plt.scatter(centroids[:,0], centroids[:,1],c='r', marker='o')
plt.title('Data appointed to 4 clusters')

#k=5
kmeans=KMeans(n_clusters=5, random_state=42)
kmeans.fit_transform(X)
y_means=kmeans.predict(X)
centroids=kmeans.cluster_centers_
colors=kmeans.labels_

#plot the blobs
plt.subplot(3,2,5)
plt.scatter(X[:,0],X[:,1],c='darkblue', edgecolor='k')
plt.scatter(centroids[:,0], centroids[:,1],c='r', marker='o')
plt.title('Unclustered data points')

#plot the clusters (k=5)
plt.subplot(3,2,6)
plt.scatter(X[:,0],X[:,1],c=colors, edgecolor='k')
plt.scatter(centroids[:,0], centroids[:,1],c='r', marker='o')
plt.title('Data appointed to 5 clusters')

plt.tight_layout()
plt.show()

'''
Conclusion: from the plots it looks like K=4 is  a better choice,despite the results from point 7
From this we can see that determining the 'correct' number of clusters is not straightforward, 
as it often involves subjective judgment.
'''

'''
#this below can be skipped

# Generate synthetic classification data
X, y_true = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0,
                                n_clusters_per_class=1, n_classes=3, random_state=42)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Compute the Voronoi diagram
vor = Voronoi(centroids)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Get consistent axis limits for all scatter plots
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Plot the true labels with Voronoi regions
colormap = cm.tab10
colors_true = colormap(y_true.astype(float) / 3)
axes[0, 0].scatter(X[:, 0], X[:, 1], c=colors_true, s=50, alpha=0.5, ec='k')
voronoi_plot_2d(vor, ax=axes[0, 0], show_vertices=False, line_colors='red', line_width=2, line_alpha=0.6, point_size=2)
axes[0, 0].set_title('Labelled Classification Data with Voronoi Regions')
axes[0, 0].set_xlabel('Feature 1')
axes[0, 0].set_ylabel('Feature 2')
axes[0, 0].set_xlim(x_min, x_max)
axes[0, 0].set_ylim(y_min, y_max)

# Call evaluate_clustering for true labels
evaluate_clustering(X, y_true, n_clusters=3, ax=axes[1, 0], title_suffix=' True Labels')

# Plot K-Means clustering results with Voronoi regions
colors_kmeans = colormap(y_kmeans.astype(float) / 3)
axes[0, 1].scatter(X[:, 0], X[:, 1], c=colors_kmeans, s=50, alpha=0.5, ec='k')
axes[0, 1].scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x', label='Centroids')
voronoi_plot_2d(vor, ax=axes[0, 1], show_vertices=False, line_colors='red', line_width=2, line_alpha=0.6, point_size=2)

axes[0, 1].set_title('K-Means Clustering with Voronoi Regions')
axes[0, 1].set_xlabel('Feature 1')
axes[0, 1].set_ylabel('Feature 2')
axes[0, 1].set_xlim(x_min, x_max)
axes[0, 1].set_ylim(y_min, y_max)

# Call evaluate_clustering for K-Means labels
evaluate_clustering(X, y_kmeans, n_clusters=3, ax=axes[1, 1], title_suffix=' K-Means Clustering')

# Adjust layout and show plot
plt.tight_layout()
plt.show()
'''
