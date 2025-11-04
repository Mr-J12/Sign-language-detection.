import cv2
import numpy as np
import os
import mediapipe as mp

# --- Setup ---
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    """ Processes an image and returns the results. """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR 2 RGB
    image.flags.writeable = False
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB 2 BGR
    return image, results

def extract_keypoints(results):
    """ Extracts keypoints from the results object. """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh]) # Total 258 features

# --- Configuration ---

# Path for exported data, numpy arrays
DATA_PATH = os.path.join(os.getcwd(), 'data', 'dynamic', 'processed_landmarks') 
RAW_VIDEO_PATH = os.path.join(os.getcwd(), 'data', 'dynamic', 'raw_videos') 

# Actions you want to detect
# !IMPORTANT: Choose 10-15 words of your choice
actions = np.array(['before', 'go', 'book', 'who', 'drink'])

# Number of videos (sequences) for each action
no_sequences = 30

# Number of frames in each sequence
sequence_length = 30

# --- Create Folders ---
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# --- Main Data Collection Loop ---
# This will iterate through your raw videos and extract features.
# You need to manually place your videos in the `raw_videos` folder
# e.g., `data/dynamic_data/raw_videos/hello/hello_1.mp4`, `.../hello_2.mp4` etc.

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Loop through actions
    for action in actions:
        video_files = [f for f in os.listdir(os.path.join(RAW_VIDEO_PATH, action)) if f.endswith('.mp4')]
        
        # Ensure we don't process more videos than no_sequences
        for sequence, video_file in enumerate(video_files[:no_sequences]):
            
            cap = cv2.VideoCapture(os.path.join(RAW_VIDEO_PATH, action, video_file))
            
            # Set frame properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Evenly sample frames across the video
            frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
            
            print(f"Processing {action} video {sequence+1}/{len(video_files[:no_sequences])}")
            
            # Loop through frames
            for frame_num in range(sequence_length):
                # Set the video to the specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[frame_num])
                
                # Read feed
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {frame_indices[frame_num]} from {video_file}. Skipping.")
                    keypoints = np.zeros(258) # Create zero array if frame is bad
                else:
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Extract keypoints
                    keypoints = extract_keypoints(results)
                
                # Export keypoints
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

            cap.release()

print("Data processing complete.")