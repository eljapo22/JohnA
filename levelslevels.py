import json
import networkx as nx
import plotly.graph_objects as go
import plotly.colors as pc
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from collections import defaultdict
from tqdm import tqdm
import os

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
        roots.add(path_data['path'][0])
    
    common_end = paths[0]['path'][-1]
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

def identify_main_paths(paths, substation_node):
    main_paths = defaultdict(list)
    for path_data in paths:
        path = path_data['path']
        if path[-1] == substation_node:
            main_paths[path[-2]].append(path)
    return main_paths

def identify_secondary_paths(paths, first_divergence_points, intersections):
    secondary_paths = defaultdict(list)
    for path_data in paths:
        path = path_data['path']
        if path[1] in first_divergence_points:
            for node in path:
                if node in intersections:
                    secondary_paths[path[1]].append(path)
                    break
    return secondary_paths

def identify_first_divergence_points(paths, substation_node):
    first_divergence_points = set()
    for path_data in paths:
        path = path_data['path']
        if path[1] != substation_node:
            first_divergence_points.add(path[1])
    return list(first_divergence_points)

def identify_hierarchy(paths, intersections, substation_node):
    G = nx.Graph()
    for path_data in paths:
        nx.add_path(G, path_data['path'])
    
    levels = {node: float('inf') for node in G.nodes()}
    levels[substation_node] = 1
    
    queue = [(substation_node, 1)]
    while queue:
        node, level = queue.pop(0)
        for neighbor in G.neighbors(node):
            if levels[neighbor] > level:
                if neighbor in intersections:
                    levels[neighbor] = level + 1
                    queue.append((neighbor, level + 1))
                else:
                    levels[neighbor] = level
                    queue.append((neighbor, level))
    
    path_hierarchy = {}
    for path_data in paths:
        path = path_data['path']
        hierarchy = []
        for i in range(len(path) - 1):
            start, end = path[i], path[i+1]
            hierarchy.append({
                'Start': start,
                'End': end,
                'Level': min(levels[start], levels[end])
            })
        path_hierarchy[path[0]] = hierarchy
    
    return path_hierarchy, max(levels.values())

def print_path_hierarchy(path_hierarchy, substation_node, first_divergence_points, main_paths, secondary_paths):
    print(f"Substation Node: {substation_node}\n")
    print(f"First Divergence Points from Substation: {', '.join(first_divergence_points)}\n")
    print("Main Paths:")
    for main_path in main_paths:
        print(f"Main Path leading to substation node {substation_node}: {main_path}")
    print("\nSecondary Paths:")
    for branch_node, paths in secondary_paths.items():
        print(f"Secondary Paths branching from {branch_node}:")
        for path in paths:
            print(f"  Path: {path}")
    print("\nHierarchy Levels for each path:")
    for path_start, hierarchy in path_hierarchy.items():
        print(f"Path starting from transformer node {path_start}:")
        for entry in hierarchy:
            print(f"  Level {entry['Level']}: Start Node: {entry['Start']}, End Node: {entry['End']}")
        print()

def closest_key_divergence_point_to_next_key_divergence_point(paths, first_divergence_points, intersections):
    key_divergence_point_connections = defaultdict(list)
    for path_data in paths:
        path = path_data['path']
        if path[1] in first_divergence_points:
            last_key_divergence_point = None
            for node in path:
                if node in intersections:
                    if last_key_divergence_point is not None:
                        key_divergence_point_connections[last_key_divergence_point].append(node)
                    last_key_divergence_point = node
    return key_divergence_point_connections

def generate_distinct_colors(num_levels):
    colors = [
        '#FF0000',  # Red for level 1
        '#006400',  # Green for level 2
        '#0000FF',  # Blue for level 3
        '#FF8C00',  # Orange for level 4
        '#FF00FF',  # Magenta for level 5
        '#000000',  # Purple for level 6
        '#800000',  # Orange for level 7
    ]
    return colors[:num_levels]

def create_figure(G, paths, intersections, roots, common_end, path_hierarchy, max_level):
    print("Creating visualization...")
    fig = go.Figure()

    distinct_colors = generate_distinct_colors(max_level)

    for path_data in paths:
        path = path_data['path']
        hierarchy = path_hierarchy[path[0]]
        for entry in hierarchy:
            start_node, end_node = entry['Start'], entry['End']
            level = entry['Level']
            x0, y0 = G.nodes[start_node]['pos']
            x1, y1 = G.nodes[end_node]['pos']
            
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                     line=dict(width=2, color=distinct_colors[level-1]),
                                     hoverinfo='text',
                                     hovertext=f"Level: {level}, Start: {start_node}, End: {end_node}"))

    node_x, node_y, node_color, node_size, node_symbol, node_text = [], [], [], [], [], []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        
        if node == common_end:
            node_color.append('red')
            node_size.append(20)
            node_symbol.append('square')
            node_text.append(node)
        elif node in roots:
            node_color.append('green')
            node_size.append(12)
            node_symbol.append('triangle-up')
            node_text.append(node)
        elif node in intersections:
            node_color.append('red')
            node_size.append(10)
            node_symbol.append('circle')
            node_text.append(node)
        else:
            node_color.append('lightblue')
            node_size.append(5)
            node_symbol.append('circle')
            node_text.append('')

    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                             marker=dict(color=node_color, size=node_size, symbol=node_symbol),
                             text=node_text,
                             textposition="top center",
                             hoverinfo='none'))

    fig.update_layout(showlegend=False, hovermode='closest',
                      margin=dict(b=20,l=5,r=5,t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    return fig

def capture_transformer_levels(path_hierarchy):
    transformer_levels = {}
    for transformer, hierarchy in path_hierarchy.items():
        transformer_levels[transformer] = hierarchy[0]['Level']
    
    sorted_transformers = sorted(transformer_levels.items(), key=lambda x: x[1])
    
    print("\nTransformer Nodes and Their Levels:")
    for transformer, level in sorted_transformers:
        print(f"Transformer: {transformer}, Level: {level}")
    
    return transformer_levels

def create_graph_database_json(G, paths_data, intersections, roots, common_end, path_hierarchy, max_level, first_divergence_points, main_paths, secondary_paths, key_divergence_point_connections, transformer_levels, input_file_path):
    graph_data = {
        "metadata": {
            "max_level": max_level,
            "common_end": common_end,
            "roots": list(roots),
            "intersections": list(intersections),
            "first_divergence_points": list(first_divergence_points)
        },
        "nodes": {},
        "relationships": [],
        "paths": {
            "main_paths": main_paths,
            "secondary_paths": secondary_paths
        },
        "key_divergence_point_connections": key_divergence_point_connections,
        "path_hierarchy": {}
    }

    for node, data in G.nodes(data=True):
        node_data = {
            "id": node,
            "type": "Regular",
            "longitude": data['pos'][0],
            "latitude": data['pos'][1],
            "shape": data.get('shape', 'circle'),
            "degree": G.degree(node)
        }
        if node in roots:
            node_data["type"] = "Transformer"
            node_data["level"] = transformer_levels.get(node)
        elif node in intersections:
            node_data["type"] = "Intersection"
            node_data["is_key_divergence_point"] = node in first_divergence_points
        elif node == common_end:
            node_data["type"] = "Substation"
        
        graph_data["nodes"][node] = node_data

    for transformer, hierarchy in path_hierarchy.items():
        graph_data["path_hierarchy"][transformer] = []
        for entry in hierarchy:
            relationship = {
                "source": entry['Start'],
                "target": entry['End'],
                "type": "CONNECTS_TO",
                "level": entry['Level']
            }
            graph_data["relationships"].append(relationship)
            graph_data["path_hierarchy"][transformer].append({
                "start": entry['Start'],
                "end": entry['End'],
                "level": entry['Level']
            })

    for path in main_paths:
        for i in range(len(path) - 1):
            for rel in graph_data["relationships"]:
                if rel["source"] == path[i] and rel["target"] == path[i+1]:
                    rel["path_type"] = "main"
                    break

    for branch_point, sec_paths in secondary_paths.items():
        for path in sec_paths:
            for i in range(len(path) - 1):
                for rel in graph_data["relationships"]:
                    if rel["source"] == path[i] and rel["target"] == path[i+1]:
                        rel["path_type"] = "secondary"
                        break

    input_file_name = os.path.basename(input_file_path)
    output_file_name = f"graph_structure_database_{os.path.splitext(input_file_name)[0]}.json"

    with open(output_file_name, 'w') as f:
        json.dump(graph_data, f, indent=2)

    print(f"Graph database structure JSON file created successfully: {output_file_name}")

def create_output_div(substation_node, first_divergence_points, main_paths, secondary_paths, path_hierarchy, key_divergence_point_connections, transformer_levels):
    output = []
    
    output.append(html.H2("Network Analysis Results"))
    
    output.append(html.H3(f"Substation Node: {substation_node}"))
    
    output.append(html.H3("First Divergence Points from Substation:"))
    output.append(html.P(', '.join(first_divergence_points)))
    
    output.append(html.H3("Main Paths:"))
    for main_path in main_paths:
        output.append(html.P(f"Main Path leading to substation node {substation_node}: {main_path}"))
    
    output.append(html.H3("Secondary Paths:"))
    for branch_node, paths in secondary_paths.items():
        output.append(html.H4(f"Secondary Paths branching from {branch_node}:"))
        for path in paths:
            output.append(html.P(f"Path: {path}"))
    
    output.append(html.H3("Hierarchy Levels for each path:"))
    for path_start, hierarchy in path_hierarchy.items():
        output.append(html.H4(f"Path starting from transformer node {path_start}:"))
        for entry in hierarchy:
            output.append(html.P(f"Level {entry['Level']}: Start Node: {entry['Start']}, End Node: {entry['End']}"))
    
    output.append(html.H3("Closest Key Divergence Point Connections:"))
    for start_node, end_nodes in key_divergence_point_connections.items():
        for end_node in end_nodes:
            output.append(html.P(f"Start Node: {start_node}, End Node: {end_node}"))
    
    output.append(html.H3("Transformer Nodes and Their Levels:"))
    sorted_transformers = sorted(transformer_levels.items(), key=lambda x: x[1])
    for transformer, level in sorted_transformers:
        output.append(html.P(f"Transformer: {transformer}, Level: {level}"))
    
    return html.Div(output)

def visualize_graph_database(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    G = nx.Graph()
    
    for node_id, node_data in data['nodes'].items():
        G.add_node(node_id, **node_data)
    
    for relationship in data['relationships']:
        G.add_edge(relationship['source'], relationship['target'], **relationship)
    
    pos = nx.spring_layout(G)
    
    edge_trace = go.Scatter(
    
        x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers', hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Type',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
    
    for node, adjacencies in enumerate(G.adjacency()):
        node_info = G.nodes[adjacencies[0]]
        node_trace['marker']['color'] += (len(adjacencies[1]),)
        node_info_text = f"ID: {adjacencies[0]}<br>Type: {node_info['type']}"
        if 'level' in node_info:
            node_info_text += f"<br>Level: {node_info['level']}"
        node_trace['text'] += (node_info_text,)
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Graph Database Visualization',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

# Main process
print("Starting main process...")
G = load_graph_from_json(r"C:\Users\eljapo22\gephi\Node2Edge2JSON\Node2Edge2JSON.json")
input_file_path = r"c:\Users\eljapo22\gephi\Middle.json"
paths_data = load_paths(input_file_path)

intersections, roots, common_end = find_intersections_roots_and_common_end(G, paths_data)
first_divergence_points = identify_first_divergence_points(paths_data, common_end)
main_paths = identify_main_paths(paths_data, common_end)
secondary_paths = identify_secondary_paths(paths_data, first_divergence_points, intersections)
path_hierarchy, max_level = identify_hierarchy(paths_data, intersections, common_end)

print_path_hierarchy(path_hierarchy, common_end, first_divergence_points, main_paths, secondary_paths)

key_divergence_point_connections = closest_key_divergence_point_to_next_key_divergence_point(paths_data, first_divergence_points, intersections)
print("\nClosest Key Divergence Point Connections:")
for start_node, end_nodes in key_divergence_point_connections.items():
    for end_node in end_nodes:
        print(f"  Start Node: {start_node}, End Node: {end_node}")

transformer_levels = capture_transformer_levels(path_hierarchy)

create_graph_database_json(G, paths_data, intersections, roots, common_end, path_hierarchy, max_level, first_divergence_points, main_paths, secondary_paths, key_divergence_point_connections, transformer_levels, input_file_path)

# Add visualization of the new JSON file
json_file_path = f"graph_structure_database_{os.path.splitext(os.path.basename(input_file_path))[0]}.json"
graph_database_fig = visualize_graph_database(json_file_path)

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Network Analysis and Visualization"),
    dcc.Tabs([
        dcc.Tab(label='Original Network', children=[
            dcc.Graph(figure=create_figure(G, paths_data, intersections, roots, common_end, path_hierarchy, max_level), id='original-graph', style={'height': '90vh'})
        ]),
        dcc.Tab(label='Graph Database', children=[
            dcc.Graph(figure=graph_database_fig, id='graph-database', style={'height': '90vh'})
        ])
    ]),
    html.Div([
        html.Button('Show Analysis Results', id='show-results-button'),
        html.Div(id='analysis-results', style={'display': 'none'})
    ])
])

@app.callback(
    Output('analysis-results', 'children'),
    Output('analysis-results', 'style'),
    Input('show-results-button', 'n_clicks')
)
def update_output(n_clicks):
    if n_clicks is None:
        return '', {'display': 'none'}
    else:
        results_div = create_output_div(common_end, first_divergence_points, main_paths, secondary_paths, path_hierarchy, key_divergence_point_connections, transformer_levels)
        return results_div, {'display': 'block'}

if __name__ == '__main__':
    print("Starting server...")
    app.run_server(debug=True)