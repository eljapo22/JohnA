import pickle

# Simulate the initialization of transformer_colors
transformer_colors = {
    1: 'rgb(255,0,0)',
    2: 'rgb(0,255,0)',
    3: 'rgb(0,0,255)',
    # Add more entries as needed
}

# Save transformer_colors dictionary
with open('C:\\Users\\eljapo22\\gephi\\cache\\transformer_colors.pkl', 'wb') as f:
    pickle.dump(transformer_colors, f)

print("transformer_colors has been saved.")
