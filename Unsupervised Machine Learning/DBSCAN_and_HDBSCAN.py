#use scikit-learn to implement DBSCAN and HDBSCAN clustering models to cultural and art facilities across Canada
#compare the performances of the two models

#Landing page:
#https://www.statcan.gc.ca/en/lode/databases/odcaf
#link to zip file:
#https://www150.statcan.gc.ca/n1/en/pub/21-26-0001/2020001/ODCAF_V1.0.zip?st=brOCT3Ry
#1.import libraries


!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install hdbscan==0.8.40
!pip install geopandas==1.0.1
!pip install contextily==1.6.2
!pip install shapely==2.0.6


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler

# geographical tools
import geopandas as gpd  # pandas dataframe-like geodataframes for geographical data
import contextily as ctx  # used for obtianing a basemap of Canada
from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')

import requests
import zipfile
import io
import os


#2. download the map of Canada
# URL of the ZIP file on the cloud server
zip_file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip'

# Directory to save the extracted TIFF file
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Download the ZIP file
response = requests.get(zip_file_url)
response.raise_for_status()  # Ensure the request was successful
# Step 2: Open the ZIP file in memory
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    # Step 3: Iterate over the files in the ZIP
    for file_name in zip_ref.namelist():
        if file_name.endswith('.tif'):  # Check if it's a TIFF file
            # Step 4: Extract the TIFF file
            zip_ref.extract(file_name, output_dir)
            print(f"Downloaded and extracted: {file_name}")

#3.create a plotting function
# Write a function that plots clustered locations and overlays them on a basemap.

def plot_clustered_locations(df,  title='Museums Clustered by Proximity'):
    """
    Plots clustered locations and overlays on a basemap.
    
    Parameters:
    - df: DataFrame containing 'Latitude', 'Longitude', and 'Cluster' columns
    - title: str, title of the plot
    """
    
    # Load the coordinates intto a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
    
    # Reproject to Web Mercator to align with basemap 
    gdf = gdf.to_crs(epsg=3857)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Separate non-noise, or clustered points from noise, or unclustered points
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]
    
    # Plot noise points 
    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')
    
    # Plot clustered points, colured by 'Cluster' number
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)
    
    # Add basemap of  Canada
    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)
    
    # Format plot
    plt.title(title, )
    plt.xlabel('Longitude', )
    plt.ylabel('Latitude', )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    # Show the plot
    plt.show()


#4. load the data and preprocess them
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df = pd.read_csv(url, encoding = "ISO-8859-1")
df=df.replace('..',np.nan) #replace the .. values with null (NaN)
df.isna().sum()#check for NaN values - most of the cols have a LOT of them
df.groupby('ODCAF_Facility_Type').count()
#or df.ODCAF_Facility_Type.value_counts()
#we only need museums
df=df[df['ODCAF_Facility_Type'] == 'museum']
#we only need Latitude and Longitude
df=df[['Latitude','Longitude']]
#we need cols to be numeric, not objects
df['Latitude']=df['Latitude'].astype('float')
df['Longitude']=df['Longitude'].astype('float')
df.dtypes
#delete null (NaN values because DBSCAN cannot work with them
df=df.dropna(subset=['Latitude','Longitude'],axis=0)

'''
#5. develop DBSCAN model
#first we need to scale the columns,in this case using standardization would be an error because we aren't using the full range of the lat/lng coordinates.
# Since latitude has a range of +/- 90 degrees and longitude ranges from 0 to 360 degrees, the correct scaling is to double the longitude coordinates (or half the Latitudes)
'''
coords_scaled = df.copy() #without this the map does not display properly, so better create a new frame instead of using df
coords_scaled["Latitude"] = 2*coords_scaled["Latitude"]

#define N and epsilon and metrics to be used
min_samples=3 # minimum number of samples needed to form a neighbourhood
eps=1.0 # neighbourhood search radius - epsilon
metric='euclidean' # distance measure 
#create DBSCAN object and fit it
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(coords_scaled)
#make prediction
dbscan.fit_predict(coords_scaled)
#store prediction (Cluster the data point belongs to) in df['Cluster']
df['Cluster']=dbscan.fit_predict(coords_scaled)
df['Cluster'].value_counts() # we have 32 clusters, and -1 which is noise. 79 points are classified as noise

#6 call the above defined function to plot the cluster on the Canada map
plot_clustered_locations(df, title='Museums Clustered by Proximity')

#7. develop HDBSCAN model
min_samples=None
min_cluster_size=3
#create HDBSCAN object and fit it
hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric='euclidean')  # You can adjust parameters as needed
#make prediction
hdb.fit_predict(coords_scaled)
df['Cluster']=hdb.fit_predict(coords_scaled)
df
df['Cluster'].value_counts() # we have a lot of clusters, and -1 which is noise. 468 points are classified as noise
plot_clustered_locations(df, title='Museums Clustered by Proximity')
