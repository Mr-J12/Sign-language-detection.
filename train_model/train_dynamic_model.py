import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_PATH = './data/dynamic/processed_landmarks/'
MODEL_SAVE_PATH = './models/dynamic_model.h5'

# Actions must match the ones from notebook 1
actions = np.array(['before', 'go', 'book', 'who', 'drink'])
NUM_CLASSES = len(actions)
# 30 frames, 258 features per frame
SEQUENCE_LENGTH = 30
FEATURES_LENGTH = 258 # 33*4 (pose) + 21*3 (left hand) + 21*3 (right hand)

# --- 1. Load Data ---
sequences, labels = [], []
for action_idx, action in enumerate(actions):
    action_path = os.path.join(DATA_PATH, action)
    sequence_dirs = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
    
    for sequence_dir in sequence_dirs:
        window = []
        sequence_path = os.path.join(action_path, sequence_dir)
        frame_files = sorted(os.listdir(sequence_path), key=lambda x: int(os.path.splitext(x)[0]))
        
        if len(frame_files) != SEQUENCE_LENGTH:
            print(f"Skipping {sequence_path}, expected {SEQUENCE_LENGTH} frames, found {len(frame_files)}")
            continue
            
        for frame_file in frame_files:
            res = np.load(os.path.join(sequence_path, frame_file))
            window.append(res)
            
        sequences.append(window)
        labels.append(action) # Use string label first

print(f"Loaded {len(sequences)} sequences.")

X = np.array(sequences)
y = np.array(labels)

# --- 2. Preprocess Labels ---
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded).astype(int)

# --- 3. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# --- 4. Build LSTM Model ---
model = Sequential()
model.add(Input(shape=(SEQUENCE_LENGTH, FEATURES_LENGTH)))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. Train Model ---
EPOCHS = 1000 # Adjust as needed
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test))

# --- 6. Evaluate and Visualize ---
print("\n--- Evaluating Model ---")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Get predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=actions))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# --- 7. Save Model ---
print(f"Saving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")