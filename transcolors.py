import pickle

# Function to clean up transformer_colors dictionary
def clean_transformer_colors(graph, transformer_colors):
    graph_nodes = set(graph.nodes())
    keys_to_delete = [node for node in transformer_colors.keys() if node not in graph_nodes]
    for key in keys_to_delete:
        print(f"Deleting node {key} from transformer_colors")  # Print statement for debugging
        del transformer_colors[key]
    return transformer_colors

# Load the graph
try:
    with open('C:\\Users\\eljapo22\\gephi\\graph_cache.pkl', 'rb') as f:
        G_proj = pickle.load(f)
    print("Graph loaded successfully.")
except FileNotFoundError:
    print("Graph cache file not found.")
    exit(1)

# Load transformer_colors dictionary
try:
    with open('C:\\Users\\eljapo22\\gephi\\cache\\transformer_colors.pkl', 'rb') as f:
        transformer_colors = pickle.load(f)
    print("Transformer colors loaded successfully.")
except FileNotFoundError:
    print("Transformer colors file not found.")
    exit(1)

# Clean the transformer_colors dictionary
transformer_colors = clean_transformer_colors(G_proj, transformer_colors)

# Save the cleaned transformer_colors dictionary
with open('C:\\Users\\eljapo22\\gephi\\cache\\transformer_colors.pkl', 'wb') as f:
    pickle.dump(transformer_colors, f)
    print("Cleaned transformer colors saved successfully.")

# Print the cleaned transformer_colors dictionary to verify
print("Cleaned transformer colors dictionary:")
print(transformer_colors)
