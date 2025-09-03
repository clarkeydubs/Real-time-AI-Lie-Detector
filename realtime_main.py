import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from collections import Counter
import os
import sounddevice as sd
import librosa
import time
from faster_whisper import WhisperModel
import threading 
from threading import Lock
from collections import deque

## Configurations 

# 6 Core CPU (8Threads)
torch.set_num_threads(8)


# Transcription Model Configs
transcript_text = ""
transcript_lock = threading.Lock()


model = WhisperModel("base", device="cpu", compute_type="int8")

# Device for each Prediction model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gesture Categories
categories = [
    "id", "OtherGestures", "Smile", "Laugh", "Scowl", "otherEyebrowMovement", "Frown", "Raise",
    "OtherEyeMovements", "Close-R", "X-Open", "Close-BE", "gazeInterlocutor", "gazeDown", "gazeUp",
    "otherGaze", "gazeSide", "openMouth", "closeMouth", "lipsDown", "lipsUp", "lipsRetracted",
    "lipsProtruded", "SideTurn", "downR", "sideTilt", "backHead", "otherHeadM", "sideTurnR",
    "sideTiltR", "waggle", "forwardHead", "downRHead", "singleHand", "bothHands", "otherHandM",
    "complexHandM", "sidewaysHand", "downHands", "upHands"
]

# Audio Configuration
recording_active = True  
audio_segment_duration = 5  
sample_rate = 8000
latest_audio_feat = None
audio_feat_lock = Lock() 

MAX_AUDIO_SEGMENTS = 10
audio_queue = deque(maxlen=MAX_AUDIO_SEGMENTS)
raw_waveform_queue = deque(maxlen=10)  

##############################################################
################    Real-Time Functions    ###################
##############################################################

def record_waveform_thread():
    while recording_active:
        try:
            samples = int(audio_segment_duration * sample_rate)
            recording = sd.rec(samples, samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            waveform = recording.flatten()

            if np.all(np.isfinite(waveform)):
                raw_waveform_queue.append(waveform)
            else:
                print("[AUDIO WARNING] Infinite waveform skipped.")
        except Exception as e:
            print(f"[AUDIO ERROR] Recording failed: {e}")

        time.sleep(0.05)

def feature_extraction_thread():
    processing_sample_rate = 4000
    low_quality_n_mels = 40
    max_frames = 70

    while recording_active:
        try:
            if raw_waveform_queue:
                # Take flattened waveform from queue
                waveform = raw_waveform_queue.pop()

                # Downsample & feature extraction
                from scipy.signal import resample_poly
                waveform_resampled = resample_poly(waveform, up=1, down=int(sample_rate / processing_sample_rate))

                mel_spec = librosa.feature.melspectrogram(y=waveform_resampled, sr=processing_sample_rate, n_mels=low_quality_n_mels)
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                if log_mel_spec.shape[1] > max_frames:
                    log_mel_spec = log_mel_spec[:, :max_frames]

                tensor = torch.tensor(log_mel_spec.T, dtype=torch.float32)

                # Save latest tensor as next audio feature
                with audio_feat_lock:
                    global latest_audio_feat
                    latest_audio_feat = tensor
                print(f"[AUDIO] Updated latest tensor, shape: {tensor.shape}")

            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")


def live_transcription_loop():
    global transcript_text
    duration = 3  

    while recording_active:
        print("[TRANSCRIBE] Listening...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio_flat = audio.flatten()

        try:
            # Transcribe from audio
            segments, _ = model.transcribe(audio_flat, language="en")

            with transcript_lock:
                for segment in segments:
                    transcript_text += " " + segment.text.strip()

            print("[TRANSCRIBE] Updated transcript.")
            
        except Exception as e:
            print(f"[TRANSCRIBE ERROR] {e}")

        time.sleep(0.05)



#####################################################
###################  Model Setup  ###################
#####################################################

from modeltranscript import TextClassifier
from modelgestures import MultiHotClassifier
from modelaudio import AudioClassifier
from modelfusion import LateFusionModel
import pickle

# Load vocab and label map
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

def tokenizer(text):
        return text.lower().split()

def encode(tokens, vocab):
        return [vocab.get(token, vocab["<unk>"]) for token in tokens]

label_map = {"truthful": 0, "deceptive": 1}
inv_label_map = {v: k for k, v in label_map.items()}

# Load models
text_model = TextClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=128, output_dim=2)
gesture_model = MultiHotClassifier(input_dim=len(categories) - 1)
audio_model = AudioClassifier(input_dim=40, output_dim=128)

text_model.load_state_dict(torch.load("text_model.pth"))
gesture_model.load_state_dict(torch.load("gesture_model.pth"))
audio_model.load_state_dict(torch.load("audio_model.pth"))

fusion_model = LateFusionModel(text_model, audio_model, gesture_model)
fusion_model.load_state_dict(torch.load("fusion_model.pth"))
fusion_model.eval()

fusion_model.to(device)

# MediaPipe Setup
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Detection Setup
# Track detected states
detections = {k: 0 for k in categories}
detections["id"] = "test_id"  
detected_movements = []  

# Facial History
nose_x_history = []
smoothed_nose_x = None
smoothed_left_iris_x = None
smoothed_left_iris_y = None
smoothed_right_iris_x = None
smoothed_right_iris_y = None
smoothed_eyebrow_left_y = None
smoothed_eyebrow_right_y = None

# Smoothing factor for low quality webcams
alpha = 0.01  

# CV2 Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Start Audio Capture & Feature Extraction Threads
t1 = threading.Thread(target=record_waveform_thread, daemon=True)
t2 = threading.Thread(target=feature_extraction_thread, daemon=True)

# ! UNCOMMENT WHEN ON POWERFUL PC RTX 4000 SERIES !
#t1.start()
#t2.start()

# Start Transcription Thread
transcription_thread = threading.Thread(target=live_transcription_loop, daemon=True)
transcription_thread.start()

#########################################################
###################     Main Loop     ###################
#########################################################

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Define webcam params
    h, w, _ = frame.shape
    image_mid_y = 0.5  
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    # Draw landmarks for display
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            #mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_TESSELATION)
            landmarks = face_landmarks.landmark

            # Facial landmark coordinates
            nose_x = landmarks[1].x
            raw_nose_x = landmarks[1].x
            nose_y = landmarks[1].y
            left_eye_x = landmarks[33].x
            right_eye_x = landmarks[263].x
            left_eye_y = landmarks[33].y
            right_eye_y = landmarks[263].y
            chin_y = landmarks[152].y

            # Eye detail detection
            r_upper = landmarks[386].y
            r_lower = landmarks[374].y
            l_eye_open = abs(landmarks[159].y - landmarks[145].y)
            r_eye_open = abs(r_upper - r_lower)

            # Raw iris position
            right_iris_x = landmarks[473].x
            right_iris_y = landmarks[473].y

            # Get key mouth landmarks
            inner_upper = landmarks[0].y
            inner_lower = landmarks[17].y
            top_lip_y = landmarks[13].y
            bottom_lip_y = landmarks[14].y
            nose_z = landmarks[1].z
            top_lip_z = landmarks[13].z
            bottom_lip_z = landmarks[14].z
            left_corner_x = landmarks[78].x
            right_corner_x = landmarks[308].x

            # Left eyebrow and eye
            eyebrow_left_y = landmarks[105].y
            eye_left_y = landmarks[159].y

            # Right eyebrow and eye
            eyebrow_right_y = landmarks[334].y
            eye_right_y = landmarks[386].y

            # Apply smoothing for jitter in facemesh on webcam
            if smoothed_eyebrow_left_y is None:
                smoothed_eyebrow_left_y = eyebrow_left_y
                smoothed_eyebrow_right_y = eyebrow_right_y
            else:
                smoothed_eyebrow_left_y = alpha * eyebrow_left_y + (1 - alpha) * smoothed_eyebrow_left_y
                smoothed_eyebrow_right_y = alpha * eyebrow_right_y + (1 - alpha) * smoothed_eyebrow_right_y


            # Inner brow distance (left to right)
            inner_brow_distance = abs(landmarks[65].x - landmarks[295].x)

            ####################################
            ####### EYEBROW DETECTIONS #########
            ####################################

            # Raise Detection
            if (eye_left_y - smoothed_eyebrow_left_y > 0.055) or (eye_right_y - smoothed_eyebrow_right_y > 0.055):
                detections["Raise"] = 1

            # Frown Detection 
            if (smoothed_eyebrow_left_y - eye_left_y > 0.09 or smoothed_eyebrow_right_y - eye_right_y > 0.09) or inner_brow_distance < 0.07:
                detections["Frown"] = 1

            # If neither then, otherEyebrowMovement 
            if all(detections.get(k, 0) == 0 for k in ["Raise", "Frown"]):
                detections["otherEyebrowMovement"] = 1

            ####################################
            ########  MOUTH DETECTIONS #########
            ####################################

            # Mouth Opening
            mouth_opening = abs(inner_upper - inner_lower)
            if mouth_opening > 0.04:
                detections["openMouth"] = 1
            elif mouth_opening < 0.015:
                detections["closeMouth"] = 1

            # Lip Vertical Movement
            if top_lip_y - nose_y > 0.1:
                detections["lipsDown"] = 1
            elif top_lip_y - nose_y < 0.05:
                detections["lipsUp"] = 1

            # Lip Horizontal Retraction
            mouth_width = abs(left_corner_x - right_corner_x)
            if mouth_width > 0.5:
                detections["lipsRetracted"] = 1

            # Lip Protrusion
            avg_lip_z = (top_lip_z + bottom_lip_z) / 2
            lip_protrusion = avg_lip_z - nose_z
            if lip_protrusion < -0.02:
                detections["lipsProtruded"] = 1

            ####################################
            ########## EYE DETECTIONS ##########
            ####################################

            # Apply smoothing to iris
            if smoothed_right_iris_x is None:
                smoothed_right_iris_x = right_iris_x
                smoothed_right_iris_y = right_iris_y
            else:
                smoothed_right_iris_x = alpha * right_iris_x + (1 - alpha) * smoothed_right_iris_x
                smoothed_right_iris_y = alpha * right_iris_y + (1 - alpha) * smoothed_right_iris_y

            # Use smoothed iris positions
            right_inner_x = landmarks[362].x
            right_outer_x = landmarks[263].x
            upper_lid = r_upper
            lower_lid = r_lower

            eye_width = right_outer_x - right_inner_x
            eye_height = upper_lid - lower_lid

            # Gaze horizontal
            horizontal_pos = (smoothed_right_iris_x - right_inner_x) / eye_width  

            # Gaze vertical
            vertical_pos = (smoothed_right_iris_y - upper_lid) / eye_height

            # Gaze Detection 
            if 0.375 < horizontal_pos < 0.675:
                detections["gazeInterlocutor"] = 1

            if horizontal_pos <= 0.25 or horizontal_pos >= 0.75:
                detections["gazeSide"] = 1

            if vertical_pos < 0.20:
                detections["gazeUp"] = 1
            elif vertical_pos > 0.80:
                detections["gazeDown"] = 1

            # Eye Open/Close Detection
            if r_eye_open < 0.01:
                detections["Close-R"] = 1

            if l_eye_open > 0.045 and r_eye_open > 0.045:
                detections["X-Open"] = 1

            if l_eye_open < 0.01 and r_eye_open < 0.01:
                detections["Close-BE"] = 1

            if 0.01 <= l_eye_open < 0.025 or 0.01 <= r_eye_open < 0.025:
                detections["OtherEyeMovements"] = 1

            # Fallback: other gaze types
            if all(detections.get(k, 0) == 0 for k in ["gazeInterlocutor", "gazeSide", "gazeUp", "gazeDown"]):
                detections["otherGaze"] = 1

            ####################################
            #### HEAD MOVEMENT DETECTIONS ######
            ####################################

            # Nose coordinates with smoothing
            if smoothed_nose_x is None:
                smoothed_nose_x = raw_nose_x
            else:
                smoothed_nose_x = alpha * raw_nose_x + (1 - alpha) * smoothed_nose_x

            nose_x_history.append(smoothed_nose_x)
            if len(nose_x_history) > 20:
                nose_x_history.pop(0)

            if smoothed_nose_x < (left_eye_x + right_eye_x) / 2 - 0.02:
                detections["SideTurn"] = 1  # Looking more to the left 
            elif smoothed_nose_x > (left_eye_x + right_eye_x) / 2 + 0.02:
                detections["sideTurnR"] = 1  # Looking more to the right

            # Threshold that defines how far until its a "tilt"
            tilt_threshold = 0.015  
            if left_eye_y > right_eye_y + tilt_threshold:
                detections["sideTilt"] = 1
            elif right_eye_y > left_eye_y + tilt_threshold:
                detections["sideTiltR"] = 1

            if chin_y - nose_y < 0.15: 
                detections["downRHead"] = 1


            # Waggle - count how many times nose_x changes direction
            if len(nose_x_history) >= 5:
                changes = 0
                prev_diff = 0
                for i in range(1, len(nose_x_history)):
                    diff = nose_x_history[i] - nose_x_history[i - 1]
                    if diff * prev_diff < 0:  
                        changes += 1
                    if abs(diff) > 0.005:  
                        changes = 0
                    prev_diff = diff
                if changes >= 3:  
                    detections["waggle"] = 1

            ####################################
            ######### HAND DETECTIONS ##########
            ####################################

            if hand_results.multi_hand_landmarks:
                hand_count = len(hand_results.multi_hand_landmarks)
                detections["singleHand"] = 1 if hand_count == 1 else 0
                detections["bothHands"] = 1 if hand_count == 2 else 0

                current_hand_positions = []
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    #mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    wrist_y = hand_landmarks.landmark[0].y
                    wrist = hand_landmarks.landmark[0]
                    middle_base = hand_landmarks.landmark[9]
                    
                    # up/down hand detections
                    if wrist_y > image_mid_y:
                        detections["downHands"] = 1
                    if wrist_y < image_mid_y:
                        detections["upHands"] = 1

                    # sideways hand detection
                    delta_x = abs(wrist.x - middle_base.x)
                    delta_y = abs(wrist.y - middle_base.y)
                    if delta_x > delta_y:
                        detections["sidewaysHand"] = 1

                    # collect current hand positions for movement analysis
                    hand_coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    current_hand_positions.append(hand_coords)

    ###############################################
    ############## Tensor Creation  ###############
    ###############################################

    # Build gesture tensor from above detections
    gesture_tensor = torch.tensor(
        [[detections[k] for k in categories if k != "id"]],
        dtype=torch.float32
    ).to(device)

    # Get latest audio tensor from global variable
    with audio_feat_lock:
        if latest_audio_feat is not None:
            audio_input = latest_audio_feat.unsqueeze(0).to(device)
        else:
            audio_input = torch.zeros((1, 100, 40), dtype=torch.float32).to(device)


    # Build text tensor
    with transcript_lock:
        clean_text = transcript_text.strip()

    tokens = tokenizer(clean_text) if clean_text else []
    token_ids_list = encode(tokens, vocab)

    if not token_ids_list:
        token_ids_tensor = torch.zeros((1, 1), dtype=torch.long)  # fallback
    else:
        token_ids_tensor = torch.tensor([token_ids_list], dtype=torch.long)

    token_ids_tensor = token_ids_tensor.to(device)

    # Run Fusion model
    with torch.no_grad():
        output = fusion_model(token_ids_tensor, gesture_tensor, audio_input)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        label = inv_label_map[pred_idx]


    # Overlay decision on webcam
    cv2.putText(frame, f"{label.upper()} ({confidence:.2f})", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show webcam
    cv2.imshow("Webcam Detection", frame)

    # Key to stop stop all detections and close program (break loop)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
    # Limit the webcam to ~30 FPS
    time.sleep(0.03)  

cap.release()
cv2.destroyAllWindows()

# Stop Audio recordings
recording_active = False

# Stop Audio feature extration and waveform gathering Threads
#t1.join()
#t2.join()
print("[AUDIO] Stopped recording.")

# Stop transcription Thread
transcription_thread.join()
print("[TEXT] Stopped Transcribing.")

# Give final prediction.
print("[PREDICTION] Final model prediction: ", label.upper())
