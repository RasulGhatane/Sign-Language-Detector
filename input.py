import cv2
import numpy as np 
import os
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from datetime import datetime
import re
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and memory growth is enabled")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
            sequence_path = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(sequence_path, exist_ok=True)
            
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
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
                npy_path = os.path.join(sequence_path, f"{frame_num}.npy")
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
            print(f"Skipping {action}: Not a directory")
            continue
        
        for sequence in range(NO_SEQUENCES):
            sequence_dir = os.path.join(action_dir, str(sequence))
            if not os.path.exists(sequence_dir):
                print(f"Warning: Missing sequence directory for {action}, sequence {sequence}")
                continue
            
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(sequence_dir, f"{frame_num}.npy")
                if not os.path.exists(npy_path):
                    print(f"Warning: Missing frame {frame_num} for {action}, sequence {sequence}")
                    break
                res = np.load(npy_path)
                window.append(res)
            
            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(actions.index(action))
            else:
                print(f"Skipping incomplete sequence: {action}, sequence {sequence}")
    
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

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

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
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
        
        history = model.fit(
            X_train, y_train,
            epochs=500,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[tensorboard_callback, early_stopping, reduce_lr, model_checkpoint]
        )
        
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        
        model.save('sign_language_model.keras')
        print("Model saved as 'sign_language_model.keras'")
        
        plot_training_history(history)
        plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
        
        print("Model training completed successfully.")
    except Exception as e:
        print(f"An error occurred during model training: {str(e)}")
        print("Please check your data and try again.")

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*&]', '', filename).replace(' ', '_')

def main():
    while True:
        action = input("Enter 'collect' to collect data, 'train' to train the model, or 'quit' to exit: ").lower()
        
        if action == 'collect':
            action_name = input("Enter the name of the sign language gesture: ")
            sanitized_name = sanitize_filename(action_name)
            sequence_count = int(input("Enter the number of sequences to collect: "))
            
            action_dir = os.path.join(DATA_PATH, sanitized_name)
            os.makedirs(action_dir, exist_ok=True)
            
            collect_data_for_action(sanitized_name, sequence_count)
            print(f"Data collection for '{action_name}' completed.")
        
        elif action == 'train':
            train_model()
        
        elif action == 'quit':
            break
        
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()