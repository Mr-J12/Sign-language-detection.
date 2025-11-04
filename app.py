import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

# --- 1. Settings & Model Loading ---
st.set_page_config(layout="wide")

# Define labels
# !IMPORTANT: Update these to match your datasets
STATIC_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
DYNAMIC_CLASSES = np.array(['before', 'go', 'book', 'who', 'drink'])
SEQUENCE_LENGTH = 30
FEATURES_LENGTH = 258

# Load Models (use caching for efficiency)
@st.cache_resource
def load_all_models():
    """ Load static and dynamic models """
    static_model_path = 'models/static_model.h5'
    dynamic_model_path = 'models/dynamic_model.h5'
    
    static_model = load_model(static_model_path)
    dynamic_model = load_model(dynamic_model_path)
    
    holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    return static_model, dynamic_model, holistic_model

try:
    static_model, dynamic_model, holistic_model = load_all_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.error("Please make sure the 'static_model.h5' and 'dynamic_model.h5' files exist in the 'models/' directory.")
    st.stop()


# --- 2. MediaPipe Helper Functions ---
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """ Processes an image and returns the results. """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """ Draws landmarks with custom styling """
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    """ Extracts keypoints from the results object. """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


# --- 3. Streamlit Video Transformer for Dynamic Detection ---
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.current_prediction = ""
        self.threshold = 0.7 # Confidence threshold

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        
        # Make detections
        image, results = mediapipe_detection(image, holistic_model)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-SEQUENCE_LENGTH:] # Keep last 30 frames
        
        if len(self.sequence) == SEQUENCE_LENGTH:
            res = dynamic_model.predict(np.expand_dims(self.sequence, axis=0))[0]
            
            if res[np.argmax(res)] > self.threshold:
                self.current_prediction = DYNAMIC_CLASSES[np.argmax(res)]
        
        # Display prediction on frame
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, self.current_prediction, (10,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return image

# --- 4. Main App Logic ---
def run_app():
    st.title("Sign Language Detection Project")
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a feature", ["Home", "Static Sign Detection", "Real-time Word Detection"])

    if app_mode == "Home":
        st.header("Welcome!")
        st.write("This application demonstrates two sign language detection models:")
        st.markdown("""
        * **Static Sign Detection:** Upload an image of a sign language letter (ASL Alphabet) for classification.
        * **Real-time Word Detection:** Use your webcam to recognize dynamic sign language words in real-time.
        """)
        st.write("Use the sidebar to navigate between features.")

    elif app_mode == "Static Sign Detection":
        st.header("Static Sign (Letter) Detection")
        st.write("Upload an image of an ASL letter (A-Z, space, del, nothing).")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            
            # Preprocess the image
            img_array = np.array(image)
            img_resized = cv2.resize(img_array, (128, 128))
            
            # Handle grayscale images
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            
            # Ensure 3 channels
            if img_resized.shape[2] == 4: # RGBA
                img_resized = img_resized[:,:,:3]

            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Make prediction
            try:
                prediction = static_model.predict(img_batch)
                predicted_class = STATIC_CLASSES[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                st.success(f"**Prediction:** {predicted_class}")
                st.info(f"**Confidence:** {confidence:.2f}%")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    elif app_mode == "Real-time Word Detection":
        st.header("Real-time Word Detection")
        st.write("Allow webcam access to start detecting dynamic signs.")
        st.warning("This model is trained on a limited vocabulary: " + ", ".join(DYNAMIC_CLASSES))
        
        webrtc_streamer(
            key="dynamic-detection",
            video_transformer_factory=SignLanguageTransformer,
            media_stream_constraints={"video": True, "audio": False}
        )

# --- 5. Time-Gated Execution ---
now = datetime.datetime.now().time()
start_time = datetime.time(18, 0) # 6 PM
end_time = datetime.time(22, 0)   # 10 PM
run_app()