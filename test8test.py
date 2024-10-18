import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import numpy as np
from shapely.geometry import Point, Polygon
import pickle
import dash

# Load the network graph
with open("C:\\Users\\eljapo22\\gephi\\network_graph.pkl", 'rb') as f:
    G = pickle.load(f)

# Substation coordinates
SUBSTATION_COORDS = (-79.4181225, 43.6789429)

def find_nearest_node(G, coordinates):
    return min(G.nodes(), key=lambda n: ((n[0]-coordinates[0])**2 + (n[1]-coordinates[1])**2))

def distribute_transformers(G, num_transformers=100):
    edges = list(G.edges())
    total_length = sum(np.sqrt((e[0][0] - e[1][0])**2 + (e[0][1] - e[1][1])**2) for e in edges)
    spacing = total_length / num_transformers
    
    transformers = []
    current_length = 0
    for edge in edges:
        edge_length = np.sqrt((edge[0][0] - edge[1][0])**2 + (edge[0][1] - edge[1][1])**2)
        while current_length + edge_length > spacing and len(transformers) < num_transformers:
            ratio = (spacing - current_length) / edge_length
            point = (edge[0][0] * (1-ratio) + edge[1][0] * ratio,
                     edge[0][1] * (1-ratio) + edge[1][1] * ratio)
            nearest_node = find_nearest_node(G, point)
            if nearest_node not in transformers:
                transformers.append(nearest_node)
            current_length = 0
            spacing = total_length / (num_transformers - len(transformers))
        current_length += edge_length
    
    return transformers

# Find the substation node and distribute transformers
substation_node = find_nearest_node(G, SUBSTATION_COORDS)
transformers = distribute_transformers(G)

# Initialize cluster data
clusters = {}
cluster_counter = 0

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='network-graph', style={'height': '90vh'}),
    html.Button('Activate Lasso', id='lasso-button'),
    html.Button('Clear Selection', id='clear-button'),
    html.Button('Save Cluster', id='save-button'),
    html.Div(id='cluster-info'),
    dcc.Store(id='lasso-data'),
    dcc.Store(id='cluster-data')
])

def create_figure():
    node_x, node_y = zip(*G.nodes())
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = edge[0]
        x1, y1 = edge[1]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                            marker=dict(size=5, color='lightgray'))
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'))
    
    transformer_trace = go.Scatter(
        x=[t[0] for t in transformers],
        y=[t[1] for t in transformers],
        mode='markers',
        marker=dict(size=10, color='yellow'),
        name='Transformers'
    )
    
    substation_trace = go.Scatter(
        x=[substation_node[0]],
        y=[substation_node[1]],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Substation'
    )
    
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        dragmode='lasso'
    )
    
    return go.Figure(data=[edge_trace, node_trace, transformer_trace, substation_trace], layout=layout)

@app.callback(
    Output('network-graph', 'figure'),
    [Input('lasso-button', 'n_clicks'),
     Input('clear-button', 'n_clicks'),
     Input('save-button', 'n_clicks')],
    [State('network-graph', 'selectedData'),
     State('cluster-data', 'data')],
    prevent_initial_call=True
)
def update_graph(lasso_clicks, clear_clicks, save_clicks, selected_data, cluster_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    fig = create_figure()
    
    if button_id == 'lasso-button':
        fig.update_layout(dragmode='lasso')
    elif button_id == 'clear-button':
        return fig
    elif button_id == 'save-button' and selected_data:
        global cluster_counter
        cluster_counter += 1
        selected_points = selected_data['points']
        selected_transformers = [transformers[p['pointIndex']] for p in selected_points 
                                 if p['curveNumber'] == 2]  # Transformer trace is the 3rd trace (index 2)
        clusters[f'Cluster_{cluster_counter}'] = selected_transformers
        
        # Highlight selected transformers
        x = [t[0] for t in selected_transformers]
        y = [t[1] for t in selected_transformers]
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', 
                                 marker=dict(size=12, color='green', symbol='star'),
                                 name=f'Cluster_{cluster_counter}'))
    
    return fig

@app.callback(
    Output('cluster-info', 'children'),
    [Input('save-button', 'n_clicks')],
    prevent_initial_call=True
)
def update_cluster_info(n_clicks):
    cluster_info = "Cluster Information:\n"
    for cluster_id, transformer_nodes in clusters.items():
        cluster_info += f"{cluster_id}: {transformer_nodes}\n"
    return cluster_info

if __name__ == '__main__':
    print(f"Substation node: {substation_node}")
    print(f"Number of transformer nodes: {len(transformers)}")
    app.run_server(debug=True)