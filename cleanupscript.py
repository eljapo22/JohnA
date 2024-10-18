import pickle

# Function to clean up transformer_colors dictionary
def clean_transformer_colors(graph, transformer_colors):
    # Get the list of nodes in the graph
    graph_nodes = set(graph.nodes())

    # Identify keys in transformer_colors that are not in graph_nodes
    keys_to_delete = [node for node in transformer_colors.keys() if node not in graph_nodes]

    # Remove these keys from transformer_colors
    for key in keys_to_delete:
        del transformer_colors[key]

    return transformer_colors

# Load the graph
with open('C:\\Users\\eljapo22\\gephi\\graph_cache.pkl', 'rb') as f:
    G_proj = pickle.load(f)

# Load transformer_colors dictionary
with open('C:\\Users\\eljapo22\\gephi\\transformer_colors.pkl', 'rb') as f:
    transformer_colors = pickle.load(f)

# Clean the transformer_colors dictionary
transformer_colors = clean_transformer_colors(G_proj, transformer_colors)

# Save the cleaned transformer_colors dictionary
with open('C:\\Users\\eljapo22\\gephi\\transformer_colors.pkl', 'wb') as f:
    pickle.dump(transformer_colors, f)

# Print the cleaned transformer_colors dictionary to verify
print(transformer_colors)
