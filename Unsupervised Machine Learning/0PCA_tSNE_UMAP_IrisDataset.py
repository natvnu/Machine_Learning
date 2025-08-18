#Apply PCA tSNE and UMAP to feature space dimensionality reduction problems
#compare visually
'''
#1. install and import libraries
%pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install matplotlib==3.9.3
!pip install plotly==5.24.1
!pip install umap-learn==0.5.7
'''
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import umap.umap_ as UMAP 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#2. load the Iris dataset
iris = load_iris()
#create iris dataframe
iris_df=pd.DataFrame(iris.data, columns=iris.feature_names)
#iris_df['species'] = iris.target_names[iris.target]


#3. Apply the PCA dimensionality reduction, from 4 to 2 features
x = iris_df
y= iris.target
#scale x
x_scaled=StandardScaler().fit_transform(x)
pca=PCA(n_components=2)
x_pca=pca.fit_transform(x_scaled) #from 4 down to 2 features
#evaluate
print(pca.explained_variance_ratio_) #[0.72962445 0.22850762]
print('Variance explained by 2 PCA: ', 100*pca.explained_variance_ratio_.sum()) #95.81320720000164


#4. Apply t-SNE dimensionality reduction, from 4 to 2 features
tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(x_scaled)
#evaluate
#t-SNE does not provide an explained_variance_ratio_ like PCA because it doesn't optimize for variance preservation. 
#Instead, t-SNE focuses on preserving local neighborhood structures in a nonlinear way, making variance metrics meaningless for its output.
#Since t-SNE is primarily used for visualization, we typically evaluate it qualitatively (by checking cluster separation).


#5. Apply UMAP reduction, from 4 to 2 features
umap_reducer = UMAP.UMAP(random_state=42)
x_umap = umap_reducer.fit_transform(x_scaled)
#evaluate
#umap also does not provide an explained_variance_ratio_, so we need to visually evaluate it

# 6. Visual evaluation
plt.figure(figsize=(15, 5))

# PCA Plot
plt.subplot(1, 3, 1)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis')
plt.title("PCA (Linear)")
plt.xlabel("PC1")
plt.ylabel("PC2")

# t-SNE Plot
plt.subplot(1, 3, 2)
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap='viridis')
plt.title("t-SNE (Nonlinear)")
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")

# UMAP Plot
plt.subplot(1, 3, 3)
plt.scatter(x_umap[:, 0], x_umap[:, 1], c=y, cmap='viridis')
plt.title("UMAP (Nonlinear)")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")

plt.tight_layout()
plt.show()


