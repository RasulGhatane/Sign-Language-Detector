import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import time
import argparse
from sklearn.preprocessing import StandardScaler

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def load_model_safe(model_path):
    try:
        model = load_model(model_path)
        print(f"Successfully loaded the model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: The model file '{model_path}' was not found.")
        print("Please ensure that you have trained the model and saved it in the specified directory.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model: {str(e)}")
        exit(1)

def load_actions(data_path):
    if not os.path.exists(data_path):
        print(f"Error: The data directory '{data_path}' was not found.")
        print("Please ensure that the path to your MP_Data directory is correct.")
        exit(1)

    actions = [action for action in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, action))]
    if not actions:
        print(f"Error: No action directories found in '{data_path}'.")
        print("Please make sure you have collected data for at least one sign language gesture.")
        exit(1)

    print(f"Loaded {len(actions)} actions: {', '.join(actions)}")
    return np.array(actions)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    # Pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    # Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    # Right Hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    def normalize_coordinates(coords, image_width, image_height):
        return [
            coord.x / image_width,
            coord.y / image_height,
            coord.z
        ]

    image_height, image_width, _ = image.shape

    pose = np.array([normalize_coordinates(res, image_width, image_height) + [res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([normalize_coordinates(res, image_width, image_height) for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([normalize_coordinates(res, image_width, image_height) for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([normalize_coordinates(res, image_width, image_height) for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}", (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def main(args):
    model = load_model_safe(args.model_path)
    actions = load_actions(args.data_path)
    
    sequence = []
    sentence = []
    predictions = deque(maxlen=args.sequence_length)
    
    colors = [(245,117,16), (117,245,16), (16,117,245)]
    colors = colors * (len(actions) // 3 + 1)
    colors = colors[:len(actions)]

    cap = cv2.VideoCapture(args.camera)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    scaler = StandardScaler()
    
    prev_time = 0
    
    with mp_holistic.Holistic(min_detection_confidence=args.min_detection_confidence, min_tracking_confidence=args.min_tracking_confidence) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            try:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-args.sequence_length:]
                
                if len(sequence) == args.sequence_length:
                    input_data = np.expand_dims(sequence, axis=0)
                    
                    input_data_flat = input_data.reshape((input_data.shape[0], -1))
                    input_data_normalized = scaler.fit_transform(input_data_flat)
                    input_data = input_data_normalized.reshape(input_data.shape)
                    
                    if np.isnan(input_data).any() or np.isinf(input_data).any():
                        print("Warning: Input data contains NaN or Inf values. Skipping prediction.")
                        continue
                    
                    res = model.predict(input_data)[0]
                    predictions.append(np.argmax(res))
                    
                    if len(predictions) == args.sequence_length:
                        most_common = max(set(predictions), key=predictions.count)
                        if predictions.count(most_common) >= args.prediction_threshold:
                            if res[most_common] > args.confidence_threshold:
                                if len(sentence) > 0:
                                    if actions[most_common] != sentence[-1]:
                                        sentence.append(actions[most_common])
                                else:
                                    sentence.append(actions[most_common])
                    
                    if len(sentence) > args.max_sentence_length:
                        sentence = sentence[-args.max_sentence_length:]
                    
                    image = prob_viz(res, actions, image, colors)
            except Exception as e:
                print(f"An error occurred during prediction: {str(e)}")
                continue
            
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, f"FPS: {fps:.2f}", (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Sign Language Detection', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Detection")
    parser.add_argument("--model_path", type=str, default="sign_language_model.keras", help="Path to the trained model")
    parser.add_argument("--data_path", type=str, default="MP_Data", help="Path to the data directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument("--sequence_length", type=int, default=30, help="Sequence length for prediction")
    parser.add_argument("--prediction_threshold", type=int, default=25, help="Threshold for prediction count")
    parser.add_argument("--confidence_threshold", type=float, default=0.8, help="Confidence threshold for prediction")
    parser.add_argument("--max_sentence_length", type=int, default=5, help="Maximum number of words in the sentence")
    parser.add_argument("--min_detection_confidence", type=float, default=0.8, help="Minimum detection confidence for MediaPipe")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.8, help="Minimum tracking confidence for MediaPipe")
    args = parser.parse_args()
    
    main(args)