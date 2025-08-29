#This code clusters the Australian cities according to the lat and long, using HDBSCAN alghoritm and plotting the clusters and noise
#it uses the AUCities.csv as datasource, not all of the cities from the Final project are in csv, but most of them are
#Clustering did a decent job, clustering Sale, Watsonia, Melbourn Airport and Melbourne in the same cluster,
#The person who wrote the lab manually grouped Watsonia, Melbourn Airport and Melbourne in the same cluster.

#1.install and import libraries
'''
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install seaborn
!pip install hdbscan
!pip install hdbscan
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan  # This is the correct import

# 2. Load dataset
df_raw = pd.read_csv('AUCities.csv', encoding='windows-1252') 
# Clean the dataframe
df = df_raw.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6'], axis=1)
df = df.dropna()
df['Long'] = df['Long'].astype('float')

# 3. Scale the coordinates properly
coords = df[['Lat', 'Long']].copy()
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# 4. Develop the HDBSCAN model
#min_cluster_size = 3
#min_samples = 2
#metric = 'euclidean'
#cluster_selection_epsilon = 0.1  # Helps merge nearby clusters
# Create and fit HDBSCAN
hdb = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size, 
    min_samples=min_samples,
    metric=metric,
    cluster_selection_epsilon=cluster_selection_epsilon
)

df['Cluster'] = hdb.fit_predict(coords_scaled)
df = df.sort_values('Cluster')

# 5. plot the results
# Prepare data for plotting
non_noise = df[df['Cluster'] != -1]
noise = df[df['Cluster'] == -1]

# Create a colormap for the clusters
unique_clusters = sorted(non_noise['Cluster'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
# plot
plt.figure(figsize=(12, 8))

# Plot each cluster with a unique color and label
for i, cluster_id in enumerate(unique_clusters):
    cluster_data = non_noise[non_noise['Cluster'] == cluster_id]
    plt.scatter(cluster_data['Long'], cluster_data['Lat'], 
                color=colors[i], s=80, edgecolor='k', alpha=0.7,
                label=f'Cluster {cluster_id}')

# Plot noise points
if len(noise) > 0:
    plt.scatter(noise['Long'], noise['Lat'], 
                color='black', s=50, edgecolor='k', alpha=0.6,
                label='Noise', marker='x')

plt.title('Australian Cities Clustered by Geographic Location (HDBSCAN)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.tight_layout()
plt.show()
df
