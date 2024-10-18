import json
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html
from collections import defaultdict
from tqdm import tqdm

def load_graph_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    print("Loading nodes and edges...")
    for node in tqdm(data['nodes'], desc="Loading nodes"):
        G.add_node(node['id'], pos=(float(node['longitude']), float(node['latitude'])), shape=node.get('shape', 'circle'))
    for edge in tqdm(data['edges'], desc="Loading edges"):
        G.add_edge(edge['source'], edge['target'])
    return G

def load_paths(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def find_intersections_roots_and_common_end(G, paths):
    subgraph_nodes = set()
    roots = set()
    for path_data in paths:
        subgraph_nodes.update(path_data['path'])
        roots.add(path_data['path'][0])  # First node is a root
    
    common_end = paths[0]['path'][-1]  # Assume the last node of the first path is the common end
    for path_data in paths[1:]:
        if path_data['path'][-1] != common_end:
            common_end = None
            break
    
    subgraph = G.subgraph(subgraph_nodes)
    
    intersections = set()
    print("Identifying intersections, roots, and common end...")
    for node in tqdm(subgraph.nodes(), desc="Checking nodes"):
        if subgraph.degree(node) > 2:
            intersections.add(node)
    
    return list(intersections), list(roots), common_end

def create_figure(G, paths, intersections, roots, common_end):
    print("Creating visualization...")
    fig = go.Figure()

    print("Adding edges...")
    edge_x, edge_y = [], []
    for edge in tqdm(G.edges(), desc="Edges"):
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'))

    print("Adding paths...")
    for path_data in tqdm(paths, desc="Paths"):
        path = path_data['path']
        path_x, path_y = [], []
        for node in path:
            x, y = G.nodes[node]['pos']
            path_x.append(x)
            path_y.append(y)
        fig.add_trace(go.Scatter(x=path_x, y=path_y, line=dict(width=2, color='lightblue'), hoverinfo='none', mode='lines'))

    print("Adding nodes...")
    node_x, node_y, node_color, node_size, node_symbol = [], [], [], [], []
    for node in tqdm(G.nodes(), desc="Nodes"):
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        if node == common_end:
            node_color.append('red')
            node_size.append(20)  # Bigger size for the common end node
            node_symbol.append('square')
        elif node in roots:
            node_color.append('green')
            node_size.append(12)
            node_symbol.append('triangle-up')
        elif node in intersections:
            node_color.append('red')
            node_size.append(10)
            node_symbol.append('circle')
        else:
            node_color.append('lightblue')
            node_size.append(5)
            node_symbol.append('circle')

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(color=node_color, size=node_size, symbol=node_symbol),
        text=[f"Node: {node}" for node in G.nodes()],
    ))

    fig.update_layout(showlegend=False, hovermode='closest',
                      margin=dict(b=20,l=5,r=5,t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    return fig

# Main process
print("Starting main process...")
G = load_graph_from_json(r"C:\Users\eljapo22\gephi\Node2Edge2JSON\Node2Edge2JSON.json")
paths_data = load_paths(r"C:\Users\eljapo22\gephi\BottomLeft.json")

intersections, roots, common_end = find_intersections_roots_and_common_end(G, paths_data)

print(f"Total number of nodes: {G.number_of_nodes()}")
print(f"Total number of paths: {len(paths_data)}")
print(f"Number of identified intersections: {len(intersections)}")
print(f"Number of root nodes: {len(roots)}")
print(f"Common end node: {common_end}")

# Create Dash app
print("Creating Dash app...")
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Network Map with Identified Intersections, Roots, and Common End"),
    dcc.Graph(figure=create_figure(G, paths_data, intersections, roots, common_end), id='graph', style={'height': '90vh'})
])

if __name__ == '__main__':
    print("Starting server...")
    app.run_server(debug=True)