
#Customer Segmentation with k-means into groups of individuals that have similar characteristics.

#install and import libraries
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
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
import plotly.express as px
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



#import and preprocess dataset
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")
cust_df.dtypes #we have one categorical column - 'Address'
cust_df['Address'].value_counts()
#we could one-hot-encode it, but in this case we will drop it 
cust_df=cust_df.drop(['Address'], axis=1)
#we will also drop all rows with null values from the dataset
cust_df = cust_df.dropna()
cust_df.isnull().sum()

#normalize the dataset to help the algorithm interpret features with different magnitudes (transform to 0 mean and standard deviation of 1)
x = StandardScaler().fit_transform(cust_df) #creates np array

#develop the clustering model
k_means = KMeans(init = "k-means++", #k-means++: Selects initial cluster centres for k-means clustering in a smart way to speed up convergence.
                 n_clusters = 3, #the number of clusters 
                 n_init = 12) #the number of times the k-means algorithm will be run with different centroid seed
#fit the model
k_means.fit(x)
#store labels (assigned clusters) in k_means_labels_
k_means_labels = k_means.labels_
k_means_labels
#store labels (assigned clusters) in k_means_cluster_centers
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers
#evaluate the model
print(f"Inertia: {k_means.inertia_:.3f}")
score = silhouette_score(x, k_means_labels)
print(f"Silhouette Score: {score:.3f}")

'''
#create new column in original df with labels
#!!!IMPORTANT!!!comment the line below for the scatter plot using NORMALIZED NP.ARRAY and k_means_labels
cust_df["Clus_km"] = k_means_labels 

#plot the distribution of customers based on their education, age and income
#2D scatter plot with Age on the x-axis, Income on the y-axis, 
#marker size representing Edu and colors representing lables (cluster asignement) 

#scatter plot using ORIGINAL DATAFRAME with added labels
plt.figure(figsize=(10, 6))

# Create scatter plot
scatter = sns.scatterplot(
    data=cust_df,
    x='Age',
    y='Income',
    size='Edu',  # Marker size represents education
    sizes=(20, 200),  # Adjust size range as needed
    hue='Clus_km',      # Color by class labels
    #palette='viridis', # Choose a color palette
    alpha=0.7         # Some transparency for better visibility
)

# Add title and labels
plt.title('Customer Distribution by Age, Income, and Education', pad=20)
plt.xlabel('Age')
plt.ylabel('Income')

# Adjust legend
plt.legend(
)

# Add a grid for better readability
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
'''

'''
#scatter plot using NORMALIZED NP.ARRAY and k_means_labels
#create dataframe out of np arrays to be able to use sns.scatterplot
column_names = cust_df.columns.tolist()
plot_df=pd.DataFrame(x, columns=column_names)
plot_df['Clus_km'] = k_means_labels 

plt.figure(figsize=(10, 6))

# Create scatter plot
scatter = sns.scatterplot(
    data=plot_df,
    x='Age',
    y='Income',
    size='Edu',  # Marker size represents education
    sizes=(20, 200),  # Adjust size range as needed
    hue='Clus_km',      # Color by class labels
    palette='viridis', # Choose a color palette
    alpha=0.7         # Some transparency for better visibility
)

# Add title and labels
plt.title('Customer Distribution by Age, Income, and Education', pad=20)
plt.xlabel('Age')
plt.ylabel('Income')

# Adjust legend
plt.legend(
)

# Add a grid for better readability
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

'''

# Create interactive 3D scatter plot using NORMALIZED NP.ARRAY and k_means_labels
fig = px.scatter_3d(plot_df, x='Edu', y='Age', z='Income', opacity=0.7, color=k_means_labels.astype(float))

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Education'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ))  # Remove color bar, resize plot

fig.show()


