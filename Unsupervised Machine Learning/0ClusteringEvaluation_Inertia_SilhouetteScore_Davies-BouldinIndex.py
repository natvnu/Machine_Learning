
#Customer Segmentation with k-means evaluation. 
#Inertia (Elbow Method) #Silhouette Score #Davies-Bouldin Index

#1. install and import libraries
'''
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install plotly
!pip install seaborn
'''
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
import plotly.express as px
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



#2. import and preprocess dataset
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")
cust_df.dtypes #we have one categorical column - 'Address'
cust_df['Address'].value_counts()
#we could one-hot-encode it, but in this case we will drop it 
cust_df=cust_df.drop(['Address'], axis=1)
#we will also drop all rows with null values from the dataset
cust_df = cust_df.dropna()
cust_df.isnull().sum()

#3. develop the clustering model

#normalize the dataset to help the algorithm interpret features with different magnitudes (transform to 0 mean and standard deviation of 1)
x = StandardScaler().fit_transform(cust_df) #creates np array

# Range of k values to test
k_values = range(2, 15)

# Store performance metrics
inertia_values = []
silhouette_scores = []
davies_bouldin_indices = []

#perform pca
pca = PCA(n_components=2) #dimensionality reduction from 9 to 2
x_pca = pca.fit_transform(x) 


for k in k_values:
    kmeans = KMeans(init = "k-means++", #k-means++: Selects initial cluster centres for k-means clustering in a smart way to speed up convergence.
                n_clusters = k) #the number of clusters
    y_kmeans = kmeans.fit_predict(x_pca)
    
    # Calculate and store metrics
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(x_pca, y_kmeans))
    davies_bouldin_indices.append(davies_bouldin_score(x_pca, y_kmeans))


print('Inertia: ', inertia_values)
print('Silhouette scores: ',  silhouette_scores)  
print('Davies Bouldin Indices: ', davies_bouldin_indices)


# Plot the inertia values (Elbow Method) #measures cluster compactnes, the lower the better
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method: Inertia vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

# Plot silhouette scores #measures separation and compactness, the higher the better
plt.subplot(1, 3, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
#Silhouette Score (Higher is Better, Range: -1 to 1):
#> 0.7: Strong structure
#0.5-0.7: Reasonable structure
#0.3-0.5: Weak structure
#< 0.2: No structure


# Plot Davies-Bouldin Index #measures cluster separation, the lower the better
plt.subplot(1, 3, 3)
plt.plot(k_values, davies_bouldin_indices, marker='o')
plt.title('Davies-Bouldin Index vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')

#CONCLUSION
#Inertia (Elbow Method): a rough idea of where the "natural" *k* might be. It's a good starting point.

#Silhouette Score: Your primary guide. It gives the most holistic view of cluster quality by balancing cohesion and separation. 
#Choose the *k* that maximizes this score.

#Davies-Bouldin Index: A great secondary check, especially to confirm the findings of the Silhouette Score. 
#Choose the *k* that minimizes this index.

#k=2 (best overall clusters (compact and separated) highest Silhouette score) or k=8(high separation - low DBI, while Silhouette score still ok)
