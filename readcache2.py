import pickle
import networkx as nx

def find_bounding_box(pkl_path):
    with open(pkl_path, 'rb') as f:
        G = pickle.load(f)

    min_lon = float('inf')
    max_lon = float('-inf')
    min_lat = float('inf')
    max_lat = float('-inf')

    for node in G.nodes():
        lon, lat = node
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)

    bounding_box = {
        "southwest": (min_lon, min_lat),
        "northeast": (max_lon, max_lat)
    }

    return bounding_box

pkl_file = r"C:\Users\eljapo22\gephi\network_graph.pkl"
bbox = find_bounding_box(pkl_file)

print("Bounding Box:")
print(f"Southwest corner: {bbox['southwest']}")
print(f"Northeast corner: {bbox['northeast']}")

width = bbox['northeast'][0] - bbox['southwest'][0]
height = bbox['northeast'][1] - bbox['southwest'][1]
print(f"\nDimensions:")
print(f"Width: {width} degrees longitude")
print(f"Height: {height} degrees latitude")