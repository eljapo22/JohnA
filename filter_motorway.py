import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from sklearn.cluster import DBSCAN
import numpy as np

# List of GeoJSON files from different expansions
geojson_files = [
    'C:/Users/eljapo22/gephi/toronto_downtown_expansion_0.geojson',
    'C:/Users/eljapo22/gephi/toronto_downtown_expansion_1.geojson',
    'C:/Users/eljapo22/gephi/toronto_downtown_expansion_2.geojson',
    'C:/Users/eljapo22/gephi/toronto_downtown_expansion_3.geojson',
    'C:/Users/eljapo22/gephi/toronto_downtown_expansion_4.geojson'
]

# Combine GeoJSON files into a single GeoDataFrame
combined_gdf = pd.concat([gpd.read_file(file) for file in geojson_files], ignore_index=True)
combined_gdf = gpd.GeoDataFrame(combined_gdf, crs=combined_gdf.crs)

# Save the combined GeoDataFrame to a new GeoJSON file
combined_geojson_path = 'C:/Users/eljapo22/gephi/combined_toronto_downtown.geojson'
combined_gdf.to_file(combined_geojson_path, driver='GeoJSON')

print(f'Combined GeoJSON data has been saved to {combined_geojson_path}')

# Function to identify and place transformers
def place_transformers(gdf, distance_threshold=500):
    transformer_points = []
    
    for _, row in gdf.iterrows():
        line = row.geometry
        highway_type = row['highway'] if 'highway' in row else None
        if isinstance(line, LineString) and highway_type not in ['motorway', 'motorway_link']:
            start_point = Point(line.coords[0])
            end_point = Point(line.coords[-1])
            transformer_points.append(start_point)
            transformer_points.append(end_point)
            
            num_points = int(line.length // distance_threshold)
            for i in range(1, num_points):
                transformer_points.append(Point(line.interpolate(i / num_points, normalized=True).coords[0]))
    
    transformer_points = list(set(transformer_points))
    transformer_gdf = gpd.GeoDataFrame(geometry=transformer_points, crs=gdf.crs)
    
    return transformer_gdf

# Place transformers on the combined GeoDataFrame
transformer_gdf = place_transformers(combined_gdf, distance_threshold=500)

# Function to eliminate clusters of transformers that are too close
def eliminate_clusters(transformer_gdf, min_distance=50):
    coords = np.array([(point.x, point.y) for point in transformer_gdf.geometry])
    db = DBSCAN(eps=min_distance / 100000, min_samples=1).fit(coords)
    labels = db.labels_
    
    unique_labels = set(labels)
    new_transformer_points = []
    
    for label in unique_labels:
        points_in_cluster = coords[labels == label]
        centroid = points_in_cluster.mean(axis=0)
        new_transformer_points.append(Point(centroid))
    
    new_transformer_gdf = gpd.GeoDataFrame(geometry=new_transformer_points, crs=transformer_gdf.crs)
    return new_transformer_gdf

# Eliminate clusters of transformers that are too close
cleaned_transformer_gdf = eliminate_clusters(transformer_gdf, min_distance=50)

# Add transformers to the original GeoDataFrame
gdf_with_transformers = gpd.GeoDataFrame(pd.concat([combined_gdf, cleaned_transformer_gdf], ignore_index=True))

# Plot the combined street grid and transformers
plt.figure(figsize=(10, 10))
base = combined_gdf.plot(color='blue', linewidth=0.5)
cleaned_transformer_gdf.plot(ax=base, marker='o', color='red', markersize=5, label='Transformers')
plt.legend()
plt.title("Combined Street Grid with Transformers")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
