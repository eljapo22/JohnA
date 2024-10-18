import json
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from collections import Counter

def load_path_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    main_paths = data['paths']['main_paths']
    
    # Include non-regular nodes and the specific node 'N-000020490'
    nodes = {k: v for k, v in data['nodes'].items() if v['type'] != 'Regular' or k == 'N-000020490'}
    
    relationships = data['relationships']
    
    print(f"Loaded {len(nodes)} nodes (including non-regular and 'N-000020490')")
    print(f"Loaded {len(relationships)} relationships")
    print(f"Loaded {len(main_paths)} main paths")
    
    # Extract secondary paths from relationships
    secondary_paths = {}
    for edge in relationships:
        if edge.get('path_type') == 'secondary':
            path_id = f"secondary_{edge['source']}_{edge['target']}"
            secondary_paths[path_id] = [edge['source'], edge['target']]
    
    print(f"Extracted {len(secondary_paths)} secondary paths")
    
    return main_paths, secondary_paths, nodes, relationships

def create_path_subgraph(main_paths, secondary_paths, nodes, relationships):
    G = nx.Graph()
    
    # Add all non-regular nodes
    for node_id, node_data in nodes.items():
        if isinstance(node_data, dict) and 'type' in node_data and node_data['type'] != 'Regular':
            G.add_node(node_id, **node_data)
            print(f"Added node: {node_id}")
        elif isinstance(node_data, dict) and 'type' not in node_data:
            print(f"Warning: Node {node_id} has no 'type' attribute: {node_data}")
            G.add_node(node_id, **node_data, type='Unknown')
            print(f"Added node: {node_id} as Unknown")
    
    # Add edges from relationships
    for edge in relationships:
        if 'source' in edge and 'target' in edge:
            if edge['source'] in G.nodes() and edge['target'] in G.nodes():
                G.add_edge(edge['source'], edge['target'], **edge)
            else:
                print(f"Warning: Edge {edge['source']} -> {edge['target']} not added. Nodes not in graph.")
    
    return G

def create_plotly_graph(G, main_paths, secondary_paths):
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    for node in G.nodes():
        x, y = G.nodes[node]['longitude'], G.nodes[node]['latitude']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_info = f"ID: {node}<br>Type: {G.nodes[node]['type']}"
        node_trace['text'] += tuple([node_info])
        node_trace['marker']['color'] += tuple([G.degree(node)])

    edge_traces = []
    
    # Add main paths
    main_colors = ['purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown']
    for i, (path_name, path_nodes) in enumerate(main_paths.items()):
        color = main_colors[i % len(main_colors)]
        if isinstance(path_nodes, list) and len(path_nodes) > 0:
            if isinstance(path_nodes[0], list):
                for subpath in path_nodes:
                    edge_trace = create_edge_trace(G, subpath, color, f"Main: {path_name}")
                    edge_traces.append(edge_trace)
            else:
                edge_trace = create_edge_trace(G, path_nodes, color, f"Main: {path_name}")
                edge_traces.append(edge_trace)
    
    # Add secondary paths
    for path_id, path_nodes in secondary_paths.items():
        edge_trace = create_edge_trace(G, path_nodes, 'pink', "Secondary Path")
        edge_traces.append(edge_trace)

    fig = go.Figure(data=[node_trace] + edge_traces,
                    layout=go.Layout(
                        title='Network graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def create_edge_trace(G, path_nodes, color, name):
    edge_x = []
    edge_y = []
    for i in range(len(path_nodes) - 1):
        if path_nodes[i] in G.nodes() and path_nodes[i+1] in G.nodes():
            x0, y0 = G.nodes[path_nodes[i]]['longitude'], G.nodes[path_nodes[i]]['latitude']
            x1, y1 = G.nodes[path_nodes[i+1]]['longitude'], G.nodes[path_nodes[i+1]]['latitude']
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        else:
            print(f"Warning: Node {path_nodes[i]} or {path_nodes[i+1]} not in graph for path {name}")
    return go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color=color),
        hoverinfo='none',
        mode='lines',
        name=name)

# Main execution
file_path = "graph_structure_database_TopLeft.json"  # Update this path if necessary
main_paths, secondary_paths, nodes, relationships = load_path_data(file_path)
G = create_path_subgraph(main_paths, secondary_paths, nodes, relationships)

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='network-graph'),
    html.Div([
        html.H4("Node Type Statistics:"),
        html.Pre(id='node-stats')
    ]),
    html.Div([
        html.H4("Path Statistics:"),
        html.Pre(id='path-stats')
    ])
])

@app.callback(
    [Output('network-graph', 'figure'),
     Output('node-stats', 'children'),
     Output('path-stats', 'children')],
    Input('network-graph', 'relayoutData'))
def update_graph(relayoutData):
    fig = create_plotly_graph(G, main_paths, secondary_paths)
    
    node_type_count = Counter(data['type'] for _, data in G.nodes(data=True))
    node_stats = "\n".join([f"{node_type}: {count}" for node_type, count in node_type_count.items()])
    
    path_stats = f"Number of main paths: {len(main_paths)}\nNumber of secondary paths: {len(secondary_paths)}"
    
    return fig, node_stats, path_stats

if __name__ == '__main__':
    app.run_server(debug=True)