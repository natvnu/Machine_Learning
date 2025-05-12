# Unupervised machine learning algorithms - clustering

The repository contains files:

- PCA _in_use.py - Principal Component Analysis (PCA) for feature space dimensionality reduction 
- Evaluation_of_kmeans.py - evaluation of the clustering model: inertia value, silhouette scores and the Davies-Bouldin index
- Kmeans_generated_data_and_customer_segmentation.py - K-Means Clustering on generated dataset and on real dataset used for customer segmentation
- DBSCAN_and_HDBSCAN.py - DBSCAN and HDBSCAN for clustering cultural and art facilities in Canada (and a zipfile with the .tiff map of Canada)

Dataset Sources: 

  - generated dataset ChurnData.cvs - historical customer dataset
  - iris dataset, available through sklearn
  - generated dataset and Cust_Segmentation.cvs - dataset provided by IBM Developer Skills Network
  - The Open Database of Cultural and Art Facilities (ODCAF), available at https://www150.statcan.gc.ca/n1/en/pub/21-26-0001/2020001/ODCAF_V1.0.zip?st=brOCT3Ry
  
Technologies Used: python, pandas, matplotlib, scikit-learn, plotly, seaborn, geopandas, contextily, shapely

Installation: copy and run the code in Jupyter Notebooks or other Python editor of choice. Keep dataset files in the same folder.

![LinearlyCorrelatedDatProjectedontoPrincipalComponents](https://github.com/natvnu/Machine_Learning/blob/main/Unsupervised%20Machine%20Learning/Linearly%20Correlated%20Data%20Projected%20onto%20Principal%20Components.png?raw=true)

![PCA 2-dimensional reduction of IRIS dataset](https://github.com/natvnu/Machine_Learning/blob/main/Unsupervised%20Machine%20Learning/PCA%202-dimensional%20reduction%20of%20IRIS%20dataset.png?raw=true)

![blobs, clusters and silhouette plot](https://github.com/natvnu/Machine_Learning/blob/main/Unsupervised%20Machine%20Learning/blobs,%20clusters%20and%20silhouette%20plot.png?raw=true
)

![Unclustered blobs and the clustering results for k = 3, 4, and 5](https://github.com/natvnu/Machine_Learning/blob/main/Unsupervised%20Machine%20Learning/Unclustered%20blobs%20and%20the%20clustering%20results%20for%20k%20=%203,%204,%20and%205.png?raw=true)

![Museums clusters in Canada](https://github.com/natvnu/Machine_Learning/blob/main/Unsupervised%20Machine%20Learning/Museums%20clusters%20in%20Canada.png?raw=true)





