
#Use Principal Component Analysis (PCA) to project 2-D data onto its principal axes
#Use PCA for feature space dimensionality reduction
#Relate explained variance to feature importance and noise reduction


#1. install and import libraries
!pip install numpy==2.2.0
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
'''

#2. PART I - use PCA to transform your 2-D data to represent it in terms of its principal axes - the projection of your data onto the two orthogonal directions that explain most of the variance in your data

#2.a. create a 2-dim dataset
np.random.seed(42)
#define mean
mean = [0, 0]
#define covariance
cov = [[3, 2], [2, 2]]
X = np.random.multivariate_normal(mean=mean, cov=cov, size=200)
X.shape
#2.b. visualise the relationship
# Scatter plot of the two features
plt.figure()
plt.scatter(X[:,0],X[:,1], edgecolor='k', alpha=0.7)
plt.title("Scatter Plot of Bivariate Normal Distribution")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis('equal')
plt.grid(True)
plt.show()

#2.c. Perform PCA dimension reduction on the dataset
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
#get the principal components
components = pca.components_
components
pca.explained_variance_ratio_# array([0.9111946, 0.0888054]), which means 91% od variance is explained by X[0], and 8.8% by X[1]

#2.d. project the data onto its principal component axes
projection_pc1 = np.dot(X, components[0])
projection_pc2 = np.dot(X, components[1])
#2.e. The new coordinates are given by the dot products of each point's coordinates with the given PCA component.
#Specifically, the projections are given by:
x_pc1 = projection_pc1 * components[0][0]
y_pc1 = projection_pc1 * components[0][1]
x_pc2 = projection_pc2 * components[1][0]
y_pc2 = projection_pc2 * components[1][1]

#2.f. Plot the original and projected data
#Plot original data
plt.figure() 
plt.scatter(X[:, 0], X[:, 1], label='Original Data', ec='k', s=50, alpha=0.6)
# Plot the projections along PC1 and PC2
plt.scatter(x_pc1, y_pc1, c='r', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 1')
plt.scatter(x_pc2, y_pc2, c='b', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 2')
plt.title('Linearly Correlated Data Projected onto Principal Components', )
plt.xlabel('Feature 1',)
plt.ylabel('Feature 2',)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
#Conclusion: The data varies in two main directions. 
#The first direction, in red, is aligned in the direction having the widest variation.
#The second direction, in blue, is perpendicular to first and has a lower variance.
'''


#3. Part II - use PCA for feature space dimensionality reduction
#3.a. Load and preprocess the Iris dataset
iris=datasets.load_iris()
x=iris.data #originally we have 4 features and 150 observations
y=iris.target
target_names=iris.target_names
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)



'''
#3.b. Build a PCA model and reduce the Iris data set dimensionality to two components
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)#now we have 2 features and 150 observations
pca.components_#principal components - np array of 4 features and 2 observations

#3.c. Plot the PCA-transformed data in 2D
plt.figure(figsize=(8,6))
colors = ['navy', 'turquoise', 'darkorange'] #because we have 3 target values (3 clusters, or 3 types of irises)
lw = 1
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    #plot a scatter plot of: x_pca first feature and x_pca second feature AND y is 0, then 1 and then 2 
    plt.scatter(x_pca[y == i, 0], x_pca[y == i, 1], color=color, s=50, ec='k',alpha=0.7, lw=lw,
                label=target_name)
plt.title('PCA 2-dimensional reduction of IRIS dataset',)
plt.xlabel("PC1",)
plt.ylabel("PC2",)
plt.legend(loc='best', shadow=False, scatterpoints=1,)
# plt.grid(True)
plt.show()

#3.d. what percentage of the original feature space variance do these two combined principal components explain?
#pca.explained_variance_ratio_
100*pca.explained_variance_ratio_.sum() #95.81

'''

#3.e. redevelop the model without reducing dimensions
pca = PCA()
x_pca = pca.fit_transform(x_scaled)#now we have 4 features and 150 observations
pca.components_#principal components - np array of 4 features and 4 observations

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio for each component
plt.figure(figsize=(10,6))
plt.bar(x=range(1, len(explained_variance_ratio)+1), height=explained_variance_ratio, alpha=1, align='center', label='PC explained variance ratio' )
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Explained Variance by Principal Components')


'''
# Plot cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.step(range(1, 5), cumulative_variance, where='mid', linestyle='--', lw=3,color='red', label='Cumulative Explained Variance')
# Only display integer ticks on the x-axis
plt.xticks(range(1, 5))
plt.legend()
plt.grid(True)
'''

'''
plt.show()
explained_variance_ratio
'''
