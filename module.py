import os
import json
import logging
from typing import Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLanguageRecognizer:
    """
    Object-oriented wrapper for sign-language video prediction using
    MediaPipe Holistic for keypoints and a Keras model for classification.
    """

    def __init__(
        self,
        model_path: str,
        mapping_path: str,
        sequence_length: int = 30,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = model_path
        self.mapping_path = mapping_path
        self.sequence_length = sequence_length
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # MediaPipe objects
        self.mp_holistic = mp.solutions.holistic
        self.holistic = None  # will be created when needed

        # Model and label maps
        self.model: Optional[tf.keras.Model] = None
        self.action_to_label = {}
        self.label_to_action = {}

        # constants
        self.feature_vector_len = 258  # 33*4 + 21*3 + 21*3

        self._load_model_and_mapping()

    # ---------------------------
    # Loading utilities
    # ---------------------------
    def _load_model_and_mapping(self):
        """Load TF model and label mapping file; sets model and mappings."""
        logger.info("Loading model and mapping...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"Mapping file not found at: {self.mapping_path}")

        # Load model
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.exception("Failed to load model.")
            raise

        # Load mapping
        try:
            with open(self.mapping_path, "r") as f:
                self.action_to_label = json.load(f)
            # invert mapping
            self.label_to_action = {int(v): k for k, v in self.action_to_label.items()}
            logger.info(f"Loaded {len(self.label_to_action)} actions from {self.mapping_path}")
        except Exception as e:
            logger.exception("Failed to load mapping file.")
            raise

    # ---------------------------
    # MediaPipe helper methods
    # ---------------------------
    def _ensure_holistic(self):
        """Create MediaPipe Holistic instance if not already created."""
        if self.holistic is None:
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    @staticmethod
    def _mediapipe_detection(image: np.ndarray, holistic) -> Tuple[np.ndarray, object]:
        """
        Convert BGR to RGB, run MediaPipe process, convert back to BGR.
        Returns processed image and the results object.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return image_bgr, results

    @staticmethod
    def _extract_keypoints(results) -> np.ndarray:
        """
        Extract keypoints from MediaPipe results into a flat vector of length 258.
        If some landmarks are missing, zeros are used.
        """
        # pose: 33 landmarks, each (x,y,z,visibility) => 33*4 = 132
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33 * 4)

        # left hand: 21 landmarks, each (x,y,z) => 21*3 = 63
        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        else:
            lh = np.zeros(21 * 3)

        # right hand: 21 landmarks
        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        else:
            rh = np.zeros(21 * 3)

        return np.concatenate([pose, lh, rh])

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        """
        Normalize single 258-length keypoint vector relative to nose (pose keypoint 0).
        If the nose point is (0,0) (i.e., not detected), it returns the original frame.
        Keeps z and visibility values unchanged except for pose x/y being anchored.
        """
        try:
            if frame.size != 258:
                # Unexpected shape; return as-is
                return frame

            # Reshape segments
            pose = frame[0:132].reshape(33, 4)
            lh = frame[132:195].reshape(21, 3)
            rh = frame[195:258].reshape(21, 3)

            anchor_x = pose[0, 0]
            anchor_y = pose[0, 1]

            if anchor_x == 0 and anchor_y == 0:
                return frame  # nothing to normalize

            norm_pose = pose.copy()
            norm_lh = lh.copy()
            norm_rh = rh.copy()

            # Shift pose x,y
            norm_pose[:, 0] = norm_pose[:, 0] - anchor_x
            norm_pose[:, 1] = norm_pose[:, 1] - anchor_y

            # Shift hands x,y
            norm_lh[:, 0] = norm_lh[:, 0] - anchor_x
            norm_lh[:, 1] = norm_lh[:, 1] - anchor_y

            norm_rh[:, 0] = norm_rh[:, 0] - anchor_x
            norm_rh[:, 1] = norm_rh[:, 1] - anchor_y

            normalized_frame = np.concatenate([norm_pose.flatten(), norm_lh.flatten(), norm_rh.flatten()])
            return normalized_frame
        except Exception:
            # If anything goes wrong, return original frame
            return frame

    # ---------------------------
    # Video processing & prediction
    # ---------------------------
    def process_video(self, video_path: str) -> np.ndarray:
        """
        Read the video and produce a (sequence_length, 258) numpy array of normalized frames.
        If video is shorter than sequence_length, pads with zero-frames.
        """
        self._ensure_holistic()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        sequence = []
        frame_count = 0

        while cap.isOpened() and frame_count < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break

            _, results = self._mediapipe_detection(frame, self.holistic)
            keypoints = self._extract_keypoints(results)
            normalized = self._normalize_frame(keypoints)
            sequence.append(normalized)
            frame_count += 1

        cap.release()

        # Pad if necessary
        if len(sequence) == 0:
            # no frames processed -> return zeros
            sequence = [np.zeros(self.feature_vector_len) for _ in range(self.sequence_length)]
        elif len(sequence) < self.sequence_length:
            pad_frame = np.zeros_like(sequence[0])
            sequence.extend([pad_frame] * (self.sequence_length - len(sequence)))

        return np.array(sequence)

    def predict_video(self, video_path: str) -> Tuple[str, float]:
        """
        Processes the video and returns (predicted_label, confidence_score).
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        try:
            sequence = self.process_video(video_path)
        except Exception as e:
            logger.exception("Video processing failed.")
            return "Video Processing Failed", 0.0

        model_input = np.expand_dims(sequence, axis=0)  # (1, seq_len, feat_len)
        preds = self.model.predict(model_input, verbose=0)
        predicted_index = int(np.argmax(preds[0]))
        confidence = float(preds[0][predicted_index])
        predicted_word = self.label_to_action.get(predicted_index, "Unknown")

        return predicted_word, confidence

    # ---------------------------
    # Cleanup
    # ---------------------------
    def close(self):
        """Release MediaPipe resources."""
        if self.holistic:
            try:
                self.holistic.close()
            except Exception:
                pass
            self.holistic = None


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Paths (change to your actual paths)
    MODEL_PATH = "sign_language_GRU_model_76.78%"
    DATA_PATH = "processed_3_data_normalized"
    MAPPING_PATH = os.path.join(DATA_PATH, "action_labels.json")
    VIDEO_PATH = "augmented_videos/young/MVI_5156.MOV"

    recognizer = None
    try:
        recognizer = SignLanguageRecognizer(
            model_path=MODEL_PATH,
            mapping_path=MAPPING_PATH,
            sequence_length=30,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        predicted_word, confidence = recognizer.predict_video(VIDEO_PATH)
        logger.info(f"Predicted word: {predicted_word}, confidence: {confidence:.4f}")

    except Exception as e:
        logger.exception("Error running recognizer.")
    finally:
        if recognizer:
            recognizer.close()
