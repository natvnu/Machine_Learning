#PCA vs t-SNE vs UMAP for Dimensionality Reduction in Generated Dataset

'''
#1. install and import libraries
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install matplotlib==3.9.3
!pip install plotly==5.24.1
!pip install umap-learn==0.5.7
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import umap.umap_ as UMAP 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px
from sklearn.datasets import make_blobs


#2. generate data
# CLuster centers:
centers = [ [ 2, -6, -6],
            [-1,  9,  4],
            [-8,  7,  2],
            [ 4,  7,  9] ]

# Cluster standard deviations:
cluster_std=[1,1,2,3.5]

# Make the blobs and return the data and the blob labels
X, y = make_blobs(n_samples=500, centers=centers, n_features=3, cluster_std=cluster_std, random_state=42)


#display data
# Create a DataFrame for Plotly
df = pd.DataFrame(X, columns=['X', 'Y', 'Z'])
'''
# Create 3 2D scatter plots (or 1 3D plot to display points (blobs))
plt.figure(figsize=(15, 5))

# Subplot 1: X vs Y
plt.subplot(1, 3, 1)
plt.scatter(df['X'], df['Y'], c=y, cmap='viridis', alpha=0.7)
plt.title("X vs Y")
plt.xlabel("X")
plt.ylabel("Y")

# Subplot 2: X vs Z
plt.subplot(1, 3, 2)
plt.scatter(df['X'], df['Z'], c=y, cmap='viridis', alpha=0.7)
plt.title("X vs Z")
plt.xlabel("X")
plt.ylabel("Z")

# Subplot 3: Y vs Z
plt.subplot(1, 3, 3)
plt.scatter(df['Y'], df['Z'], c=y, cmap='viridis', alpha=0.7)
plt.title("Y vs Z")
plt.xlabel("Y")
plt.ylabel("Z")

plt.tight_layout()
plt.show()

'''
#scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)



#3. apply PCA dimensionality reduction, from 4 to 2 features
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled) 
#plot PCA reduced data
fig = plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis')
plt.title("PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")



#4. apply t-SNE dimensionality reduction, from 4 to 2 features
tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(x_scaled) 
#plot t-sne reduced data
plt.subplot(1, 3, 2)
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap='viridis')
plt.title("t_SNE")
plt.xlabel("PC1")
plt.ylabel("PC2")

#5. apply umap dimensionality reduction, from 4 to 2 features
umap = UMAP.UMAP(random_state=42)
x_umap = umap.fit_transform(x_scaled) 
#plot UMAP reduced data
plt.subplot(1, 3, 3)
plt.scatter(x_umap[:, 0], x_umap[:, 1], c=y, cmap='viridis')
plt.title("UMAP")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

