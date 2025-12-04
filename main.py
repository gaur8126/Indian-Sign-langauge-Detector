import cv2
import numpy as np
import json
import os
import tempfile
import traceback
import uvicorn
import tensorflow as tf
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

# --- CONFIGURATION ---
MODEL_PATH = 'sign_language_GRU_model_76.78%' 
MAPPING_PATH = 'processed_3_data_normalized/action_labels.json' 

app = FastAPI(title="ISL Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MEDIAPIPE SETUP ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- HELPER FUNCTIONS ---

def mediapipe_detection(image, model):
    """Converts image color space and processes with MediaPipe."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """Draws stylized landmarks on the image for visual feedback."""
    # Draw Left Hand
    mp_drawing.draw_landmarks(
        image, 
        results.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    # Draw Right Hand
    mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    """Extracts and flattens keypoints from results."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def normalize_frame(frame):
    """Normalizes keypoints relative to the pose anchor."""
    try:
        pose = frame[0:132].reshape(33, 4)
        lh = frame[132:195].reshape(21, 3)
        rh = frame[195:258].reshape(21, 3)
        anchor_x = pose[0, 0]
        anchor_y = pose[0, 1]
        
        # Avoid division by zero or invalid anchors
        if anchor_x == 0 and anchor_y == 0: return frame 
        
        norm_pose = pose.copy()
        norm_lh = lh.copy()
        norm_rh = rh.copy()
        
        norm_pose[:, 0] -= anchor_x
        norm_pose[:, 1] -= anchor_y
        norm_lh[:, 0] -= anchor_x
        norm_lh[:, 1] -= anchor_y
        norm_rh[:, 0] -= anchor_x
        norm_rh[:, 1] -= anchor_y
        
        return np.concatenate([norm_pose.flatten(), norm_lh.flatten(), norm_rh.flatten()])
    except:
        return frame

# --- MODEL CLASS ---

class ISLModel:
    def __init__(self):
        self.model = None
        self.label_to_action = {}
        self.load()

    def load(self):
        print("--- Loading Model System ---")
        # 1. Load Model
        if os.path.exists(MODEL_PATH):
            try:
                self.model = tf.keras.models.load_model(MODEL_PATH)
                print(f"Model Loaded Successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"WARNING: Model file not found at: {MODEL_PATH}")

        # 2. Load Mappings
        if os.path.exists(MAPPING_PATH):
            try:
                with open(MAPPING_PATH, 'r') as f:
                    action_to_label = json.load(f)
                # Ensure keys are integers
                self.label_to_action = {int(v): k for k, v in action_to_label.items()}
                print(f"Mappings Loaded: {len(self.label_to_action)} labels")
            except Exception as e:
                print(f"Error loading mappings: {e}")
        else:
            print(f"WARNING: Mapping file not found at: {MAPPING_PATH}")

    def predict(self, sequence):
        if self.model is None:
            return "Model Not Loaded", 0.0

        input_data = np.expand_dims(sequence, axis=0)
        res = self.model.predict(input_data, verbose=0)
        
        pred_idx = np.argmax(res[0])
        confidence = float(res[0][pred_idx])
        
        label = self.label_to_action.get(int(pred_idx), "Unknown")
        return label, confidence

# --- INITIALIZATION ---
isl_system = ISLModel()
camera = cv2.VideoCapture(0) # Open camera once at startup

# --- STREAMING GENERATOR ---
def generate_frames():
    """
    Reads from camera, processes frame, predicts sign, 
    draws overlay, and yields MJPEG stream.
    """
    sequence = []
    current_label = "Waiting..."
    current_conf = 0.0
    WINDOW_SIZE = 30 

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = camera.read()
            if not success:
                # If camera fails, try to reconnect or just break
                break

            # 1. MediaPipe Detection
            image, results = mediapipe_detection(frame, holistic)
            
            # 2. Draw Landmarks
            draw_styled_landmarks(image, results)
            
            # 3. Prediction Logic
            keypoints = extract_keypoints(results)
            normalized = normalize_frame(keypoints)
            
            sequence.append(normalized)
            sequence = sequence[-WINDOW_SIZE:] # Keep last 30 frames

            # Only predict if we have a full window
            if len(sequence) == WINDOW_SIZE:
                # Perform prediction
                label, conf = isl_system.predict(np.array(sequence))
                current_label = label
                current_conf = conf

            # 4. Draw User Interface on Frame
            # Create a header background
            cv2.rectangle(image, (0,0), (640, 50), (245, 117, 16), -1)
            
            # Display Prediction
            display_text = f"{current_label} ({int(current_conf*100)}%)"
            cv2.putText(image, display_text, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display collecting status if filling buffer
            if len(sequence) < WINDOW_SIZE:
                cv2.putText(image, f"Gathering: {len(sequence)}/30", (400, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            # 5. Encode Frame
            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()

            # 6. Yield Frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ROUTES ---

@app.get("/")
async def read_root():
    """Serves the index.html file."""
    if os.path.exists('index.html'):
        return FileResponse('index.html')
    return {"error": "index.html missing. Please save the HTML code as index.html"}

@app.get("/video_feed")
def video_feed():
    """Route for the live video stream."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/predict/video")
async def predict_video_endpoint(file: UploadFile = File(...)):
    """Route for the file upload tab."""
    try:
        # Save upload to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
            temp_vid.write(await file.read())
            temp_path = temp_vid.name

        cap = cv2.VideoCapture(temp_path)
        sequence = []
        TARGET_FRAMES = 30
        
        # Process video file
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened() and len(sequence) < TARGET_FRAMES:
                ret, frame = cap.read()
                if not ret: break
                
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                normalized = normalize_frame(keypoints)
                sequence.append(normalized)

        cap.release()
        if os.path.exists(temp_path): os.unlink(temp_path)

        if len(sequence) == 0:
            return JSONResponse({"error": "No frames extracted"}, status_code=400)
        
        # Pad if video is too short (duplicate last frame)
        while len(sequence) < TARGET_FRAMES:
            sequence.append(sequence[-1])

        label, conf = isl_system.predict(np.array(sequence))
        return {"label": label, "confidence": conf}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.on_event("shutdown")
def shutdown_event():
    """Clean up camera on server shutdown."""
    camera.release()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)