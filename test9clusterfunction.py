import json
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html
import numpy as np

# Load the graph from the cache
def load_graph_from_cache():
    with open("C:\\Users\\eljapo22\\gephi\\cache\\7644e20afe6c4a94db3dcb733771bb0d4d5a6fc6.json", 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    for element in data['elements']:
        if element['type'] == 'node':
            G.add_node(element['id'], pos=(float(element['lon']), float(element['lat'])))
        elif element['type'] == 'way':
            for i in range(len(element['nodes']) - 1):
                G.add_edge(element['nodes'][i], element['nodes'][i + 1])
    return G

G = load_graph_from_cache()

app = Dash(__name__)












def create_figure():
    fig = go.Figure()
    # Add all grid edges as a single trace
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(color='rgba(112, 128, 144, 0.5)', width=1),
        mode='lines',
        hoverinfo='none',
        showlegend=False
    ))

    # Define the three paths
    paths = [
        [6532307595, 24959511, 6400935136, 10040870233, 6532307581, 400958053, 5245333107, 11645252769, 6532307622, 6532307624, 1553140355, 6532307627, 393551208, 29603262, 6532307556],
        [24959509, 390543741, 2411583461, 42606772, 2122030533, 2122030527, 10086042916, 24959510, 10086042915, 286269584, 2411583460, 390539237, 6400935402, 24959511, 6532307574, 391004375, 6400935183, 6400935182, 3081114187, 42606901, 394502503, 24959515, 394502518, 6400935388, 6400935389, 6400935391, 60052055, 389678256, 21436518, 391169149, 21436517, 393546090, 60052034, 2274582844, 6369142313, 394502565, 6378669378, 29604711, 6378669367, 394502560, 80927373, 29690737, 6373975070, 42606813, 6373975072, 6373975071, 2823222757, 389677917, 6373975073, 29690555, 321508813, 389677923, 2823222758, 321508812, 24959561, 123347984, 242413453, 24959560, 389678221, 24959559, 389678039],
        [391169149, 21436517, 6400935151, 389678251, 87486207, 29605000, 42606791, 393551200, 20979760]
    ]

    colors = ['black', 'blue', 'forestgreen']
    blue_shades = ['rgba(30,144,255,1)', 'rgba(255,0,0,1)', 'rgba(0,0,0,1)', 'rgba(128,128,128,1)']  # L1 (light blue), L2 (red), L3 (black), N (grey)

    for i, path in enumerate(paths):
        path_x, path_y = [], []
        for node in path:
            if node in G.nodes:
                x, y = G.nodes[node]['pos']
                path_x.append(x)
                path_y.append(y)
            else:
                print(f"Warning: Node {node} not found in graph")

        if i == 1:  # For DS Feeder 11
            for j in range(4):  # Create 4 parallel lines
                offset = (j - 1.5) * 0.0001  # Adjust this value to change the spacing
                dx = np.diff(path_x)
                dy = np.diff(path_y)
                perp_x = -dy / np.sqrt(dx**2 + dy**2)
                perp_y = dx / np.sqrt(dx**2 + dy**2)
                
                new_x = np.array(path_x) + offset * np.concatenate(([perp_x[0]], perp_x))
                new_y = np.array(path_y) + offset * np.concatenate(([perp_y[0]], perp_y))
                
                line_name = f'DS Feeder 11 - {"N" if j == 3 else f"L{j+1}"}'
                fig.add_trace(go.Scatter(
                    x=new_x, y=new_y,
                    mode='lines',
                    line=dict(color=blue_shades[j], width=2),
                    name=line_name,
                    hoverinfo='name'
                ))
        else:  # For Path 1 and Path 3
            path_name = "DS TF BB" if i == 0 else "DS TF CC"
            for j in range(3):  # Create 3 parallel dashed lines
                offset = (j - 1) * 0.0001  # Adjust this value to change the spacing
                dx = np.diff(path_x)
                dy = np.diff(path_y)
                perp_x = -dy / np.sqrt(dx**2 + dy**2)
                perp_y = dx / np.sqrt(dx**2 + dy**2)
                
                new_x = np.array(path_x) + offset * np.concatenate(([perp_x[0]], perp_x))
                new_y = np.array(path_y) + offset * np.concatenate(([perp_y[0]], perp_y))
                
                line_color = 'forestgreen' if j == 1 else 'black'
                fig.add_trace(go.Scatter(
                    x=new_x, y=new_y,
                    mode='lines',
                    line=dict(color=line_color, width=2, dash='dash'),
                    name=f'{path_name} - Line {j+1}',
                    hoverinfo='name'
                ))

        # Add start and end nodes
        if path_x and path_y:
            start_x, start_y = path_x[0], path_y[0]
            end_x, end_y = path_x[-1], path_y[-1]
            
            # Set marker properties based on the path
            if i == 1:  # DS Feeder 11
                start_color = 'red'
                end_color = 'green'
                start_symbol = 'square'
                end_symbol = 'triangle-up'
                start_text = 'Start DS Feeder 11'
                end_text = 'End DS Feeder 11'
                start_size = 10
                end_size = 10
            elif i == 0:  # Path 1 (DS TF BB)
                start_color = 'green'
                end_color = 'navy'
                start_symbol = 'triangle-up'
                end_symbol = 'circle'
                start_text = 'DS TF BB'
                end_text = 'End DS TF BB'
                start_size = 15  # Increased size for green triangle
                end_size = 10
            else:  # Path 3 (DS TF CC)
                start_color = 'green'
                end_color = 'brown'
                start_symbol = 'triangle-up'
                end_symbol = 'circle'
                start_text = 'DS TF CC'
                end_text = 'End DS TF CC'
                start_size = 15  # Increased size for green triangle
                end_size = 10
            
            fig.add_trace(go.Scatter(
                x=[start_x, end_x],
                y=[start_y, end_y],
                mode='markers',
                marker=dict(
                    color=[start_color, end_color],
                    size=[start_size, end_size],
                    symbol=[start_symbol, end_symbol]
                ),
                text=[start_text, end_text],
                hoverinfo='text',
                name=f'Start/End {"DS Feeder 11" if i == 1 else path_name}'
            ))

        # Add additional nodes for DS TF BB (Path 1)
        if i == 0:
            additional_node_ids = [6532307624, 5245333107]
            for idx, node_id in enumerate(additional_node_ids):
                if node_id in path:
                    index = path.index(node_id)
                    fig.add_trace(go.Scatter(
                        x=[path_x[index]],
                        y=[path_y[index]],
                        mode='markers',
                        marker=dict(
                            color='navy',
                            size=10,
                            symbol='circle'
                        ),
                        text=f'Area MM Meter {idx+1}',
                        hoverinfo='text',
                        name=f'Area MM Meter {idx+1}'
                    ))
                else:
                    print(f"Warning: Additional node {node_id} not found in path")

        # Add additional nodes for DS TF CC (Path 3)
        elif i == 2:
            additional_node_ids = [42606791, 87486207]
            for idx, node_id in enumerate(additional_node_ids):
                if node_id in path:
                    index = path.index(node_id)
                    fig.add_trace(go.Scatter(
                        x=[path_x[index]],
                        y=[path_y[index]],
                        mode='markers',
                        marker=dict(
                            color='brown',
                            size=10,
                            symbol='circle'
                        ),
                        text=f'Area KK Meter {idx+3}',
                        hoverinfo='text',
                        name=f'Area KK Meter {idx+3}'
                    ))
                else:
                    print(f"Warning: Additional node {node_id} not found in path")

    fig.update_layout(showlegend=True)
    
    return fig






    

# Update the app layout
app.layout = html.Div([
    html.H1("Network Map with Highlighted Paths"),  # Edit this to change the title
    dcc.Graph(figure=create_figure(), id='graph', style={'height': '90vh'})  # Edit style to change graph size
])

if __name__ == '__main__':
    app.run_server(debug=True)