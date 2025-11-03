import os
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# -------------------------
# STEP 1: Set Paths and Parameters
# -------------------------
data_dir = r"C:\Users\lalit\Downloads\leapGestRecog\leapGestRecog"
MODEL_PATH = "gesture_mobilenetv2.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# -------------------------
# STEP 2: Data Loading
# -------------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation'
)

num_classes = len(train_gen.class_indices)
label_map = {v: k for k, v in train_gen.class_indices.items()}
print("Label Map:", label_map)

# -------------------------
# STEP 3: Build Model
# -------------------------
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------
# STEP 4: Train Model
# -------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("✅ Model trained successfully and saved as:", MODEL_PATH)

# -------------------------
# STEP 5: Real-Time Prediction (Webcam)
# -------------------------
def preprocess_frame(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def realtime_prediction():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    print("🎥 Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x = preprocess_frame(frame)
        preds = model.predict(x, verbose=0)[0]
        pred_idx = np.argmax(preds)
        gesture = list(train_gen.class_indices.keys())[pred_idx]
        confidence = preds[pred_idx]

        cv2.putText(frame, f"{gesture} ({confidence:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Uncomment to test live webcam
# realtime_prediction()
