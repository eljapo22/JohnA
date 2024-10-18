import pickle
import networkx as nx

# Path to the cache file
cache_file_path = r"C:\Users\eljapo22\gephi\network_graph.pkl"

# Load the contents of the pickle file
with open(cache_file_path, 'rb') as f:
    G = pickle.load(f)

# Print basic information about the graph
print("Graph Information:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Analyze node attributes
node_attributes = set()
for node, data in G.nodes(data=True):
    node_attributes.update(data.keys())

print("\nAll node attributes:")
print(', '.join(node_attributes))

# Print a sample of nodes with all their attributes
print("\nSample of Nodes (first 5):")
for node in list(G.nodes())[:100]:
    print(f"Node {node}:")
    for key, value in G.nodes[node].items():
        print(f"  {key}: {value}")
    print("---")

# Analyze edge attributes
edge_attributes = set()
for _, _, data in G.edges(data=True):
    edge_attributes.update(data.keys())

print("\nAll edge attributes:")
print(', '.join(edge_attributes))

# Print a sample of edges with all their attributes
print("\nSample of Edges (first 5):")
for u, v, data in list(G.edges(data=True))[:100]:
    print(f"Edge: {u} -> {v}")
    for key, value in data.items():
        print(f"  {key}: {value}")
    print("---")

# Check for graph-level attributes
print("\nGraph Attributes:")
for key, value in G.graph.items():
    print(f"{key}: {value}")

# If 'way' information exists, it might be in a different format or location
# Let's check if it's a graph-level attribute
if 'ways' in G.graph:
    print("\nWay information found in graph attributes")
    print(f"Number of ways: {len(G.graph['ways'])}")
    print("Sample of ways (first 5):")
    for way in G.graph['ways'][:100]:
        print(way)
        print("---")