import json
from collections import defaultdict

def load_combined_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_level_mapping(edge_list):
    level_map = {}
    for edge in edge_list:
        level_map[(edge['source'], edge['target'])] = str(edge['level'])
    return level_map

def find_main_paths(data, level_map):
    main_paths = []
    for path_list in data['paths']['main_paths'].values():
        for path in path_list:
            path_with_levels = []
            for i in range(len(path) - 1):
                level = level_map.get((path[i], path[i+1]), 'Unknown')
                path_with_levels.append((path[i], level))
            path_with_levels.append((path[-1], '1'))  # Assume substation is level 1
            main_paths.append(path_with_levels)
    return main_paths

def get_node_info(data, node_id):
    node_data = data['nodes'].get(node_id, {})
    node_type = node_data.get('type', 'Unknown')
    if node_type == 'Intersection':
        return f"Intersection (ID: {node_id})"
    elif node_type == 'Transformer':
        return f"Transformer (ID: {node_id})"
    elif node_type == 'Substation':
        return f"Substation (ID: {node_id})"
    return f"Regular (ID: {node_id})"




def find_last_common_level1_nodes(main_paths):
    # Collect the last level 1 node for each path, excluding the substation
    last_level1_nodes = []
    for path in main_paths:
        for node, level in reversed(path[:-1]):  # Exclude substation
            if level == '1':
                last_level1_nodes.append(node)
                break
    
    print("Last level 1 node in each path:")
    for i, node in enumerate(last_level1_nodes):
        print(f"Path {i+1}: {node}")

    if not all(last_level1_nodes):
        return None, "Not all paths have level 1 nodes"

    # Count occurrences of each last level 1 node
    node_counts = defaultdict(int)
    for node in last_level1_nodes:
        node_counts[node] += 1

    # Find nodes that appear in at least two paths
    common_nodes = [node for node, count in node_counts.items() if count >= 2]

    print(f"Common last level 1 nodes (appearing in at least two paths): {common_nodes}")

    if not common_nodes:
        return None, "No common last level 1 nodes found across paths"

    if len(common_nodes) == 1:
        return common_nodes[0], f"Found 1 common last level 1 node"
    else:
        return common_nodes, f"Found {len(common_nodes)} common last level 1 nodes"








def analyze_network(file_path):
    data = load_combined_data(file_path)
    level_map = create_level_mapping(data['edge_list'])
    main_paths = find_main_paths(data, level_map)
    last_common_level1_nodes, status = find_last_common_level1_nodes(main_paths)

    result = []
    result.append("Network Analysis Summary:")
    result.append("=" * 30)
    result.append(f"Substation Node ID: {data['metadata']['common_end']}")
    result.append(f"Number of Main Paths: {len(main_paths)}")
    result.append(f"Number of Intersections: {len(data['metadata']['intersections'])}")
    result.append(f"\nLAST COMMON LEVEL 1 NODE(S) BEFORE SUBSTATION:")
    if isinstance(last_common_level1_nodes, str):  # Single node case
        result.append(f"  - {last_common_level1_nodes} : {get_node_info(data, last_common_level1_nodes)}")
    elif isinstance(last_common_level1_nodes, list):  # Multiple nodes case
        for node in last_common_level1_nodes:
            result.append(f"  - {node} : {get_node_info(data, node)}")
    else:
        result.append(f"  {status}")
    result.append("=" * 30)


    result.append("\nDetailed Main Paths Analysis:")
    for i, path in enumerate(main_paths, 1):
        result.append(f"\nMain Path {i}:")
        result.append(f"  Start: {path[0][0]} (Root)")
        result.append(f"  End: {path[-1][0]} (Substation)")
        result.append(f"  Path Length: {len(path)} nodes")
        
        level1_nodes = [node for node, level in path if level == '1']
        result.append(f"  Number of Level 1 nodes: {len(level1_nodes)}")
        result.append(f"  Last Level 1 Node: {level1_nodes[-2] if len(level1_nodes) > 1 else 'None'}")  # Exclude substation
        result.append(f"  All Levels in Path: {', '.join(sorted(set(level for _, level in path)))}")

        intersections = [node for node, _ in path if node in data['metadata']['intersections']]
        result.append(f"  Intersections: {', '.join(intersections) if intersections else 'None'}")

    return "\n".join(result)




if __name__ == "__main__":
    file_path = r"c:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_TopLeft.json"
    network_analysis = analyze_network(file_path)
    print(network_analysis)