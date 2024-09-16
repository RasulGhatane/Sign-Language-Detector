import cv2
import numpy as np 
import os
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from datetime import datetime
import re
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("GPU is available and memory growth is enabled")
    except RuntimeError as e:
        logging.error(f"GPU error: {e}")
else:
    logging.info("No GPU found. Using CPU.")

logging.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('MP_Data')
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def collect_data_for_action(action, sequence_count):
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for sequence in range(sequence_count):
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to capture frame")
                    continue

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()

def load_data():
    sequences, labels = [], []
    actions = [action for action in os.listdir(DATA_PATH) if not action.startswith('.')]
    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.isdir(action_dir):
            logging.warning(f"Skipping {action}: Not a directory")
            continue
        
        for sequence in range(NO_SEQUENCES):
            sequence_dir = os.path.join(action_dir, str(sequence))
            if not os.path.exists(sequence_dir):
                logging.warning(f"Missing sequence directory for {action}, sequence {sequence}")
                continue
            
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(sequence_dir, f"{frame_num}.npy")
                if not os.path.exists(npy_path):
                    logging.warning(f"Missing frame {frame_num} for {action}, sequence {sequence}")
                    break
                res = np.load(npy_path)
                window.append(res)
            
            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(actions.index(action))
            else:
                logging.warning(f"Skipping incomplete sequence: {action}, sequence {sequence}")
    
    if not sequences:
        raise ValueError("No valid sequences found. Please check your data collection process.")
    
    return np.array(sequences), np.array(labels)

def preprocess_data(sequences, labels):
    n_sequences, n_steps, n_features = sequences.shape
    sequences_reshaped = sequences.reshape(n_sequences * n_steps, n_features)

    scaler = StandardScaler()
    sequences_normalized = scaler.fit_transform(sequences_reshaped)

    sequences_normalized = sequences_normalized.reshape(n_sequences, n_steps, n_features)
    
    return sequences_normalized, labels

def create_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    try:
        sequences, labels = load_data()
        sequences_normalized, labels = preprocess_data(sequences, labels)

        X_train, X_test, y_train, y_test = train_test_split(sequences_normalized, labels, test_size=0.2, random_state=42)

        input_shape = (SEQUENCE_LENGTH, sequences.shape[2])
        num_classes = len(np.unique(labels))
        model = create_model(input_shape, num_classes)

        log_dir = os.path.join('Logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

        history = model.fit(
            X_train, y_train,
            epochs=500,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[tensorboard_callback, early_stopping, reduce_lr]
        )

        test_loss, test_acc = model.evaluate(X_test, y_test)
        logging.info(f"Test accuracy: {test_acc}")

        model.save('sign_language_model.h5')
        logging.info("Model saved as 'sign_language_model.h5'")
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*&]', '', filename).replace(' ', '_')

def main():
    while True:
        action = input("Enter 'collect' to collect data, 'train' to train the model, or 'q' to quit: ").lower()
        if action == 'q':
            break
        elif action == 'collect':
            sign = input("Enter the sign you want to add: ").lower()
            sanitized_sign = sanitize_filename(sign)
            action_path = os.path.join(DATA_PATH, sanitized_sign)
            os.makedirs(action_path, exist_ok=True)
            
            sequence_count = int(input(f"How many sequences do you want to record for '{sign}'? "))
            
            for sequence in range(sequence_count):
                os.makedirs(os.path.join(action_path, str(sequence)), exist_ok=True)
            
            logging.info(f"Preparing to collect data for '{sign}'. Press 'q' to stop recording early.")
            collect_data_for_action(sanitized_sign, sequence_count)
            
            logging.info(f"Data collection for '{sign}' completed.")
        elif action == 'train':
            logging.info("Starting model training...")
            train_model()
            logging.info("Model training completed.")
        else:
            logging.warning("Invalid input. Please try again.")

if __name__ == "__main__":
    os.makedirs(DATA_PATH, exist_ok=True)
    main()