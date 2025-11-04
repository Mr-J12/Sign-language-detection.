import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuration ---
# Ensure these paths are correct and accessible
TRAIN_DIR = 'C:/Users/yashw/OneDrive/Desktop/data/static/train/'
TEST_DIR = 'C:/Users/yashw/OneDrive/Desktop/data/static/test/'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
MODEL_SAVE_PATH = './models/static_model.h5' 

# --- 1. Data Preprocessing (Using ImageDataGenerator) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', 
    color_mode='rgb' 
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', 
    shuffle=False,  # Important for confusion matrix
    color_mode='rgb' 
)

# Get class labels
class_labels = list(train_generator.class_indices.keys())
print(f"Class indices found: {train_generator.class_indices}")
NUM_CLASSES = len(class_labels)

# --- !! ADDED DEBUGGING PRINTS !! ---
# These lines will tell you immediately if your paths are wrong
print(f"\nFound {train_generator.samples} training images belonging to {NUM_CLASSES} classes.")
print(f"Found {validation_generator.samples} validation images belonging to {len(validation_generator.class_indices)} classes.")
print("---------------------------------------------------\n")
# --- !! END ADDED !! ---


# --- 2. Model Building (Transfer Learning) ---

# Load base model - MobileNetV2 expects input_shape=(height, width, 3)
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False 

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# --- 3. Model Training ---
EPOCHS = 100

# Check if validation data is available before training
if validation_generator.samples == 0:
    print("Warning: No validation data found. Skipping validation during training.")
    print(f"Please check the path: {TEST_DIR}")
    history = model.fit(
        train_generator,
        epochs=EPOCHS
    )
else:
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )


# --- 4. Model Evaluation & Visualization ---
print("\n--- Evaluating Model ---")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
if 'val_loss' in history.history:
     plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# --- 4. Model Evaluation & Visualization ---
print("\n--- Evaluating Model ---")

# ... [Your plotting code for accuracy and loss] ...
plt.show()

# --- !! UPDATED EVALUATION SECTION !! ---
# Classification Report and Confusion Matrix (only if validation data exists)
if validation_generator.samples > 0:
    print("\n--- Generating Classification Report & Confusion Matrix ---")
    
    # Calculate the number of steps needed to cover all validation samples
    # --- THIS IS THE FIXED LINE ---
    validation_steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))
    
    # Get predictions
    Y_pred = model.predict(validation_generator, steps=validation_steps)
    
    # Slice predictions to match the number of samples (in case steps * batch_size > samples)
    Y_pred = Y_pred[:validation_generator.samples]
    
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = validation_generator.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
else:
    print("\n--- Skipping Classification Report and Confusion Matrix ---")
    print(f"Reason: The validation generator found 0 samples in the directory: {TEST_DIR}")
    print("Please check the 'TEST_DIR' path and ensure it contains subdirectories for each class, with images inside.")


# --- 5. Save the Model ---
print(f"Saving model to {MODEL_SAVE_PATH}...")
# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")