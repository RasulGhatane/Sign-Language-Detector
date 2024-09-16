import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import time
import logging
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

MODEL_PATH = 'sign_language_model.h5'
DATA_PATH = r'C:\Users\rasul\Documents\Python\Sign-Language_Detector\MP_Data'
SEQUENCE_LENGTH = 30
PREDICTION_THRESHOLD = 0.8
PREDICTION_FRAMES = 25
MAX_SENTENCE_LENGTH = 5

def load_model_safe(model_path):
    try:
        model = load_model(model_path)
        logging.info(f"Successfully loaded the model from {model_path}")
        return model
    except FileNotFoundError:
        logging.error(f"The model file '{model_path}' was not found. Please ensure that you have trained the model and saved it in the same directory as this script.")
        exit(1)
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {str(e)}")
        exit(1)

def load_actions(data_path):
    if not os.path.exists(data_path):
        logging.error(f"The data directory '{data_path}' was not found. Please ensure that the path to your MP_Data directory is correct.")
        exit(1)

    actions = [action for action in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, action))]
    if not actions:
        logging.error(f"No action directories found in '{data_path}'. Please make sure you have collected data for at least one sign language gesture.")
        exit(1)

    logging.info(f"Loaded {len(actions)} actions: {', '.join(actions)}")
    return np.array(actions)

def verify_model_actions_compatibility(model, actions):
    sample_input = np.zeros((1, SEQUENCE_LENGTH, 1662)) 
    model_output = model.predict(sample_input)
    if model_output.shape[1] != len(actions):
        logging.warning(f"Model output shape ({model_output.shape[1]}) does not match the number of actions ({len(actions)})")
        logging.info("Attempting to adjust the model...")
        model = adjust_model(model, len(actions))
    logging.info("Model and actions are compatible.")
    return model

def adjust_model(model, num_actions):
    new_input = Input(shape=model.input_shape[1:]) 
    x = new_input
    for layer in model.layers[:-1]:
        x = layer(x)

    new_output = Dense(num_actions, activation='softmax')(x)

    new_model = Model(inputs=new_input, outputs=new_output)

    new_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    
    logging.info(f"Model adjusted to output {num_actions} actions.")
    return new_model

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Face mesh
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    try:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        keypoints = np.concatenate([pose, face, lh, rh])
        logging.info(f"Extracted keypoints shape: {keypoints.shape}")
        return keypoints
    except Exception as e:
        logging.error(f"Error extracting keypoints: {str(e)}")
        return np.zeros((SEQUENCE_LENGTH, 1662))  # Adjust size if needed

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}", (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def main():
    model = load_model_safe(MODEL_PATH)
    actions = load_actions(DATA_PATH)
    model = verify_model_actions_compatibility(model, actions)

    sequence = []
    sentence = []
    predictions = deque(maxlen=SEQUENCE_LENGTH)
    
    colors = [(245,117,16), (117,245,16), (16,117,245)]
    colors = colors * (len(actions) // 3 + 1)
    colors = colors[:len(actions)]

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame")
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            try:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-SEQUENCE_LENGTH:]
                
                if len(sequence) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(sequence, axis=0)
                    if np.isnan(input_data).any() or np.isinf(input_data).any():
                        logging.warning("Input data contains NaN or Inf values. Skipping prediction.")
                        continue
                    
                    logging.info(f"Input data shape: {input_data.shape}")
                    res = model.predict(input_data)[0]
                    logging.info(f"Model prediction: {res}")
                    
                    predictions.append(np.argmax(res))
                    
                    if len(predictions) == SEQUENCE_LENGTH:
                        most_common = max(set(predictions), key=predictions.count)
                        if predictions.count(most_common) >= PREDICTION_FRAMES:
                            if res[most_common] > PREDICTION_THRESHOLD:
                                if len(sentence) > 0:
                                    if actions[most_common] != sentence[-1]:
                                        sentence.append(actions[most_common])
                                else:
                                    sentence.append(actions[most_common])
                    
                    if len(sentence) > MAX_SENTENCE_LENGTH:
                        sentence = sentence[-MAX_SENTENCE_LENGTH:]
                    
                    image = prob_viz(res, actions, image, colors)
            
            except Exception as e:
                logging.error(f"Error during processing: {str(e)}")
            
            cv2.imshow('Sign Language Detection', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
