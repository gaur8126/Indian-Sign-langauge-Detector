
import os
import numpy as np
import json

def normalize_sequence(sequence):
    """
    Normalizes a single sequence (30, 258) of keypoints
    by making them relative to the nose (pose keypoint 0).
    """
    normalized_sequence = []
    for frame in sequence:
        # Reshape frame to easily access keypoints
        pose = frame[0:132].reshape(33, 4)
        lh = frame[132:195].reshape(21, 3)
        rh = frame[195:258].reshape(21, 3)
        
        # Get nose (keypoint 0) x,y as the anchor
        anchor_x = pose[0, 0]
        anchor_y = pose[0, 1]
        
        # Handle case where anchor is 0 (e.g., failed detection)
        if anchor_x == 0 and anchor_y == 0:
            normalized_sequence.append(frame)
            continue
            
        # Create new normalized keypoints
        norm_pose = pose.copy()
        norm_lh = lh.copy()
        norm_rh = rh.copy()
        
        # --- Normalize Pose ---
        norm_pose[:, 0] = norm_pose[:, 0] - anchor_x
        norm_pose[:, 1] = norm_pose[:, 1] - anchor_y
        
        # --- Normalize Hands ---
        norm_lh[:, 0] = norm_lh[:, 0] - anchor_x
        norm_lh[:, 1] = norm_lh[:, 1] - anchor_y
        
        norm_rh[:, 0] = norm_rh[:, 0] - anchor_x
        norm_rh[:, 1] = norm_rh[:, 1] - anchor_y
        
        # Flatten and concatenate back
        normalized_frame = np.concatenate([
            norm_pose.flatten(), 
            norm_lh.flatten(), 
            norm_rh.flatten()
        ])
        normalized_sequence.append(normalized_frame)
        
    return np.array(normalized_sequence)


INPUT_PATH = 'processed_2_data' 

# OUTPUT: The NEW folder where we'll save the final data
OUTPUT_PATH = 'processed_3_data_normalized'

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"Loading data from {INPUT_PATH}...")
try:
    # Load data using YOUR file names
    X_data = np.load(os.path.join(INPUT_PATH, 'X.npy'))
    y_labels = np.load(os.path.join(INPUT_PATH, 'y.npy'))
    with open(os.path.join(INPUT_PATH, 'action_labels.json'), 'r') as f:
        action_to_label = json.load(f)
        
except FileNotFoundError:
    print(f"ERROR: Could not find files in {INPUT_PATH}.")
    print("Please make sure the INPUT_PATH is correct.")
    raise

print(f"Data loaded. X shape: {X_data.shape}, y shape: {y_labels.shape}")
print(f"Found {len(action_to_label)} classes.")

# This will hold our new, normalized data
X_normalized = []

print("Starting normalization loop... (This will take a few minutes)")
for i, sequence in enumerate(X_data):
    try:
        norm_seq = normalize_sequence(sequence)
        X_normalized.append(norm_seq)
    except Exception as e:
        print(f"Error normalizing sequence {i}. Skipping. Error: {e}")
        X_normalized.append(sequence) 
        
    if (i+1) % 5000 == 0:
        print(f"  Processed {i+1} / {len(X_data)} sequences")

print("Normalization complete.")

# --- Step 5: Save the New Normalized Data ---

X_normalized_array = np.array(X_normalized)
print(f"New normalized X shape: {X_normalized_array.shape}")

print(f"Saving new files to {OUTPUT_PATH}...")
# Save files with the same naming convention as your original
np.save(os.path.join(OUTPUT_PATH, 'X.npy'), X_normalized_array)
np.save(os.path.join(OUTPUT_PATH, 'y.npy'), y_labels)
with open(os.path.join(OUTPUT_PATH, 'action_labels.json'), 'w') as f:
    json.dump(action_to_label, f)

print("--- All Done! ---")
print(f"Your new, normalized data is ready in {OUTPUT_PATH}.")