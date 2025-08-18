#PCA for feature space dimensionality reduction
#Use PCA to project the four-dimensional Iris feature data set down onto a two-dimensional feature space.

This will have the added benefit of enabling you to visualize some of the most important structures in the dataset.

#1. install and import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#2. Load dataset
iris = load_iris()
# Convert to DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]
iris_df.head()

# 3. reduce the Iris data set dimensionality to two components using PCA
#set x and y (in this case we can do it, in other cases we may not be able to do so)
x=iris_df.drop(['species'],axis=1)
y=iris_df[['species']]
#scale x
x_scaled=StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
x_scaled #4 columns
x_pca #2 columns
pca.explained_variance_ratio_ #[0.72962445, 0.22850762], 72.9% of data variance is explained by 1st PCA and 22.8% by second PCA
print('Variance explained by 2 PCA: ', 100*pca.explained_variance_ratio_.sum())

#4. reinitialize the PCA model without reducing the dimension
pca = PCA() #no n_components defined
x_pca = pca.fit_transform(x_scaled)
x_pca #4 columns
pca.explained_variance_ratio_ #[0.72962445, 0.22850762, 0.03668922, 0.00517871]
print('Variance explained by all 4 PCA: ', 100*pca.explained_variance_ratio_.sum())
#conclusion - 2 features explain 95.8% of variance in data, and all 4 features explain 99.9%

pca_df=pd.DataFrame(pca.explained_variance_ratio_) 
pca_df.plot(kind='bar', legend=False)
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Explained Variance by Principal Components')

plt.show()

