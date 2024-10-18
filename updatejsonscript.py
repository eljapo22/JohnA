import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def update_transformer(data):
    for node_id, node_data in data['nodes'].items():
        if node_data.get('display') == 'D/S T/F:EE':
            node_data['Area'] = 'EE'
            print(f"Updated transformer {node_id} with Area: EE")
            break

def update_specific_nodes(data, node_ids):
    for node_id in node_ids:
        if node_id in data['nodes']:
            data['nodes'][node_id]['type'] = 'meter'
            data['nodes'][node_id]['Area'] = 'EE'
            print(f"Updated node {node_id} to type 'meter' and added Area: EE")
        else:
            print(f"Node {node_id} not found in the data")

def main():
    file_path = r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_Middle.json"
    
    # Load the JSON data
    data = load_json(file_path)
    
    # Update the transformer
    update_transformer(data)
    
    # Update specific nodes
    nodes_to_update = ['N-000020297', 'N-000023029', 'N-000023033']

    update_specific_nodes(data, nodes_to_update)
    
    # Save the updated JSON data
    save_json(data, file_path)
    print(f"Updated JSON saved to {file_path}")

if __name__ == "__main__":
    main()