
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json


MODEL_PATH = 'sign_language_GRU_model_76.78%'

DATA_PATH = 'processed_3_data_normalized'
MAPPING_PATH = os.path.join(DATA_PATH, 'action_labels.json')




# -------------------------------------
print("Loading model and label mapping...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(MAPPING_PATH, 'r') as f:
        action_to_label = json.load(f)
    
    label_to_action = {v: k for k, v in action_to_label.items()}
    
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Loaded {len(label_to_action)} actions from {MAPPING_PATH}")
    
except Exception as e:
    print(f"ERROR: Could not load model or mapping files. Check your paths.")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Mapping Path: {MAPPING_PATH}")
    raise e

# Step 4: Define ALL Preprocessing Functions
# ------------------------------------------
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                   # image is no longer writeable
    results = model.process(image)                  # make prediction
    image.flags.writeable = True                    # image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,lh, rh])

def normalize_frame(frame):
    """
    Normalizes a SINGLE frame (258 features) of keypoints
    by making them relative to the nose (pose keypoint 0).
    """
    try:
        # Reshape frame to easily access keypoints
        pose = frame[0:132].reshape(33, 4)
        lh = frame[132:195].reshape(21, 3)
        rh = frame[195:258].reshape(21, 3)
        
        # Get nose (keypoint 0) x,y as the anchor
        anchor_x = pose[0, 0]
        anchor_y = pose[0, 1]
        
        # Handle case where anchor is 0 (e.g., failed detection)
        if anchor_x == 0 and anchor_y == 0:
            return frame 
            
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
        return normalized_frame
    except Exception as e:
        # If any error, just return the original frame
        return frame

# Function to process video and extract keypoints
def process_video(video_path, sequence_length=30):  # 30 frames = 1 second at 30fps
    # Initialize MediaPipe Holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Read the video
        cap = cv2.VideoCapture(video_path)
        sequence = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            normalized_frame = normalize_frame(keypoints)
            sequence.append(normalized_frame)
            frame_count += 1
            
        
        # If we didn't get enough frames, pad with zeros
        while len(sequence) < sequence_length:
            sequence.append(np.zeros_like(sequence[0]))
            
    return np.array(sequence)

# Step 5: Define Prediction Function
# ----------------------------------
def predict_video(video_path):
    """
    Processes a video file and returns the model's prediction.
    """
    # 1. Process the video into a (30, 258) sequence
    sequence = process_video(video_path)
    if sequence is None:
        return "Video Processing Failed", 0.0
    
    # 2. Add the batch dimension -> (1, 30, 258)
    model_input = np.expand_dims(sequence, axis=0)
    
    # 3. Get predictions
    predictions = model.predict(model_input, verbose=0)
    
    # 4. Get the top prediction
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    
    # 5. Get the word label
    predicted_word = label_to_action.get(predicted_index, "Unknown")
    
    return predicted_word, confidence


print("\n--- Starting Video Test ---")

video_path = "augmented_videos/young/MVI_5156.MOV"
predicted_word, confidence = predict_video(video_path)
print(f"predicted word : {predicted_word}, with confidence of: {confidence}")