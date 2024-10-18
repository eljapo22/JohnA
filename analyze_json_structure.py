import json
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any
from itertools import combinations

def load_combined_data(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return json.load(f)

def create_level_mapping(edge_list: List[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    return {(edge['source'], edge['target']): str(edge['level']) for edge in edge_list}

def find_main_paths(data: Dict[str, Any], level_map: Dict[Tuple[str, str], str]) -> List[List[Tuple[str, str]]]:
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

def get_node_info(data: Dict[str, Any], node_id: str) -> str:
    node_data = data['nodes'].get(node_id, {})
    node_type = node_data.get('type', 'Unknown')
    display = node_data.get('display', '')
    return f"{node_type} (ID: {node_id}, Display: {display})"

def analyze_network(file_path: str) -> str:
    data = load_combined_data(file_path)
    level_map = create_level_mapping(data['edge_list'])
    main_paths = find_main_paths(data, level_map)

    result = []
    result.append("Network Analysis Summary:")
    result.append("=" * 30)
    result.append(f"Substation Node ID: {data['metadata']['common_end']}")
    result.append(f"Number of Main Paths: {len(main_paths)}")
    result.append(f"Number of Intersections: {len(data['metadata']['intersections'])}")

    # Enhanced Path Divergence Analysis
    divergence_analysis = analyze_path_divergence(main_paths, data)
    result.append("\nEnhanced Path Divergence Analysis:")
    for level, points in divergence_analysis.items():
        result.append(f"  Level {level}: {len(points)} divergence points")
        for point, info in points.items():
            result.append(f"    Divergence point: {point}")
            result.append(f"      Occurs in {sum(info['occurrences'].values())} paths:")
            for div_level, count in info['occurrences'].items():
                transformer_counts = Counter(info['transformers'][div_level])
                transformers = [f"Transformer ({occurrences} occurrence{'s' if occurrences > 1 else ''}) {get_node_info(data, t)}" for t, occurrences in transformer_counts.items()]
                result.append(f"        - {count} path(s) at Level {div_level}")
                result.append(f"          Transformers: {', '.join(transformers)}")

    # Common Subpath Detection
    common_subpaths = find_common_subpaths(main_paths)
    result.append("\nCommon Subpath Analysis:")
    for length, subpaths in common_subpaths.items():
        if subpaths:
            result.append(f"  Length {length}: {len(subpaths)} common subpaths")
            result.append(f"    Most common: {subpaths[0]} (occurs in {len(subpaths[0][1])} paths)")

    # Level Transition Analysis
    level_transitions = analyze_level_transitions(main_paths)
    result.append("\nLevel Transition Analysis:")
    for transition, count in level_transitions.most_common(5):
        result.append(f"  {transition[0]} -> {transition[1]}: {count} occurrences")

    # Node Type Distribution
    node_types = analyze_node_types(data, main_paths)
    result.append("\nNode Type Distribution:")
    for node_type, count in node_types.items():
        result.append(f"  {node_type}: {count}")

    # Intersection Node Analysis
    intersection_analysis = analyze_intersections(main_paths, data['metadata']['intersections'])
    result.append("\nIntersection Node Analysis:")
    for intersection, info in intersection_analysis.items():
        result.append(f"  {intersection}: {get_node_info(data, intersection)}")
        result.append(f"    Occurs in {info['occurrences']} paths")
        result.append(f"    Connected to {len(info['connected_to'])} nodes")
        result.append(f"    Positions: {info['positions']}")

    # Path Complexity Metrics
    complexity_metrics = calculate_path_complexity(main_paths, data['metadata']['intersections'])
    result.append("\nPath Complexity Metrics:")
    for metric, values in complexity_metrics.items():
        result.append(f"  {metric}:")
        result.append(f"    Average: {sum(values) / len(values):.2f}")
        result.append(f"    Min: {min(values)}")
        result.append(f"    Max: {max(values)}")

    # Substation Proximity Analysis
    substation_proximity = analyze_substation_proximity(main_paths, data['metadata']['common_end'])
    result.append("\nSubstation Proximity Analysis:")
    result.append(f"  Average nodes before substation: {substation_proximity['avg_nodes']:.2f}")
    result.append(f"  Average level changes before substation: {substation_proximity['avg_level_changes']:.2f}")

    # Root Node Characteristics
    root_node_analysis = analyze_root_nodes(main_paths)
    result.append("\nRoot Node Analysis:")
    for root, info in root_node_analysis.items():
        result.append(f"  {root}:")
        result.append(f"    Connected to {len(info['connected_to'])} nodes")
        result.append(f"    Levels of connected nodes: {', '.join(map(str, info['connected_levels']))}")

    # Path Similarity Scoring
    similarity_scores = calculate_path_similarity(main_paths)
    result.append("\nPath Similarity Scoring:")
    most_similar = max(similarity_scores, key=similarity_scores.get)
    least_similar = min(similarity_scores, key=similarity_scores.get)
    result.append(f"  Most similar paths: {most_similar} (score: {similarity_scores[most_similar]:.2f})")
    result.append(f"  Least similar paths: {least_similar} (score: {similarity_scores[least_similar]:.2f})")

    # Detailed Main Paths Analysis
    result.append("\nDetailed Main Paths Analysis:")
    for i, path in enumerate(main_paths, 1):
        result.append(f"\nMain Path {i}:")
        result.append(f"  Start: {path[0][0]} ({get_node_info(data, path[0][0])})")
        result.append(f"  End: {path[-1][0]} ({get_node_info(data, path[-1][0])})")
        result.append(f"  Path Length: {len(path)} nodes")
        
        level_counts = Counter(level for _, level in path)
        result.append("  Level Distribution:")
        for level, count in sorted(level_counts.items()):
            result.append(f"    Level {level}: {count} nodes")
        
        intersections = [node for node, _ in path if node in data['metadata']['intersections']]
        result.append(f"  Intersections: {', '.join(intersections) if intersections else 'None'}")
        
        node_types = Counter(get_node_info(data, node).split()[0] for node, _ in path)
        result.append("  Node Type Distribution:")
        for node_type, count in node_types.items():
            result.append(f"    {node_type}: {count} nodes")

    return "\n".join(result)

def analyze_path_divergence(main_paths, data):
    divergence_points = defaultdict(lambda: defaultdict(lambda: {'occurrences': defaultdict(int), 'transformers': defaultdict(list)}))
    for i, path1 in enumerate(main_paths):
        for j, path2 in enumerate(main_paths[i+1:], start=i+1):
            for (node1, level1), (node2, level2) in zip(path1, path2):
                if node1 != node2:
                    div_point = divergence_points[level1][node1]
                    div_point['occurrences'][level1] += 1
                    div_point['occurrences'][level2] += 1
                    div_point['transformers'][level1].append(path1[0][0])
                    div_point['transformers'][level2].append(path2[0][0])
                    break
    return divergence_points

def find_common_subpaths(main_paths):
    common_subpaths = defaultdict(list)
    for length in range(2, 10):  # Adjust range as needed
        subpaths = defaultdict(set)
        for i, path in enumerate(main_paths):
            for j in range(len(path) - length + 1):
                subpath = tuple(node for node, _ in path[j:j+length])
                subpaths[subpath].add(i)
        common = [(subpath, paths) for subpath, paths in subpaths.items() if len(paths) > 1]
        common_subpaths[length] = sorted(common, key=lambda x: len(x[1]), reverse=True)
    return common_subpaths

def analyze_level_transitions(main_paths):
    transitions = Counter()
    for path in main_paths:
        levels = [level for _, level in path]
        transitions.update(zip(levels, levels[1:]))
    return transitions

def analyze_node_types(data, main_paths):
    node_types = Counter()
    for path in main_paths:
        for node, _ in path:
            node_type = get_node_info(data, node).split()[0]
            node_types[node_type] += 1
    return node_types

def analyze_intersections(main_paths, intersections):
    intersection_analysis = {intersection: {'occurrences': 0, 'positions': [], 'connected_to': set()} for intersection in intersections}
    
    for path_index, path in enumerate(main_paths):
        for node_index, (node, _) in enumerate(path):
            if node in intersections:
                intersection_analysis[node]['occurrences'] += 1
                intersection_analysis[node]['positions'].append((path_index, node_index))
                if node_index > 0:
                    intersection_analysis[node]['connected_to'].add(path[node_index-1][0])
                if node_index < len(path) - 1:
                    intersection_analysis[node]['connected_to'].add(path[node_index+1][0])
    
    return intersection_analysis

def calculate_path_complexity(main_paths, intersections):
    metrics = {
        'level_changes': [],
        'intersection_count': [],
        'level1_ratio': []
    }
    for path in main_paths:
        levels = [level for _, level in path]
        metrics['level_changes'].append(sum(1 for a, b in zip(levels, levels[1:]) if a != b))
        metrics['intersection_count'].append(sum(1 for node, _ in path if node in intersections))
        metrics['level1_ratio'].append(sum(1 for _, level in path if level == '1') / len(path))
    return metrics

def analyze_substation_proximity(main_paths, substation_id):
    nodes_before_substation = []
    level_changes_before_substation = []
    for path in main_paths:
        substation_index = next(i for i, (node, _) in enumerate(path) if node == substation_id)
        nodes_before_substation.append(len(path) - substation_index)
        levels = [level for _, level in path[substation_index:]]
        level_changes_before_substation.append(sum(1 for a, b in zip(levels, levels[1:]) if a != b))
    return {
        'avg_nodes': sum(nodes_before_substation) / len(nodes_before_substation),
        'avg_level_changes': sum(level_changes_before_substation) / len(level_changes_before_substation)
    }

def analyze_root_nodes(main_paths):
    root_nodes = {}
    for path in main_paths:
        root, root_level = path[0]
        if root not in root_nodes:
            root_nodes[root] = {'connected_to': set(), 'connected_levels': set()}
        root_nodes[root]['connected_to'].add(path[1][0])
        root_nodes[root]['connected_levels'].add(path[1][1])
    return root_nodes

def calculate_path_similarity(main_paths):
    similarity_scores = {}
    for (i, path1), (j, path2) in combinations(enumerate(main_paths), 2):
        shared_nodes = set(node for node, _ in path1) & set(node for node, _ in path2)
        similarity = len(shared_nodes) / max(len(path1), len(path2))
        similarity_scores[(i, j)] = similarity
    return similarity_scores

if __name__ == "__main__":
    file_path = r"c:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_TopLeft.json"
    network_analysis = analyze_network(file_path)
    print(network_analysis)