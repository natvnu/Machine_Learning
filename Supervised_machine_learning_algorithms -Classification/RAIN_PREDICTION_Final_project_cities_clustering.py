#This code clusters the Australian cities according to the lat and long, using HDBSCAN alghoritm and plotting the clusters and noise
#it uses the AUCities.csv as datasource, not all of the cities from the Final project are in csv, but most of them are
#Clustering did a decent job, clustering Sale, Watsonia, Melbourn Airport and Melbourne in the same cluster,
#The person who wrote the lab manually grouped Watsonia, Melbourn Airport and Melbourne in the same cluster.

#1.import libraries

!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install seaborn
!pip install hdbscan

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN

#2. load dataset
df_raw=pd.read_csv('AUCities.csv',encoding='windows-1252') #add this encoding or it won't work

df=df_raw.drop(['City','Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6'], axis=1)
df=df.dropna()
df['Long']=df['Long'].astype('float')

#3. develop the HDBSCAN model
#scale the coordinates, otherwise everything will be clustered as noise
coords_scaled=df.copy()
coords_scaled['Lat']=2*coords_scaled['Lat']

#define n and metric
min_samples=3
metric='euclidean'
#create HDBSCAN object and fit it
hdb=HDBSCAN(min_cluster_size=3, min_samples=min_samples,metric=metric).fit(coords_scaled)
#make prediction
df['Cluster']=hdb.fit_predict(coords_scaled)

df=df.sort_values(by='Long')
df['City']=df_raw.sort_values(by='Long')['City']
df=df.sort_values(by='Cluster') #this is the df of cities with clusters they should belong to

#plot the cities on
non_noise=df[df['Cluster']!=-1]
noise=df[df['Cluster']==-1]

plt.figure(figsize=(10,5))
plt.scatter(non_noise['Lat'], non_noise['Long'],c=non_noise['Cluster'], s=50, cmap='tab10', edgecolor='k', alpha=0.6, label='Clusters') #not exactly correct display in legend
plt.scatter(noise['Lat'], noise['Long'], color='black', s=50, edgecolor='k', alpha=0.6,label='Noise')
plt.title('Australian cities grouped in clusters')
plt.xlabel('Latitude')
plt.xlabel('Longitude')
plt.legend()
plt.tight_layout()
plt.show()

df
