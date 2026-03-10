import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time

# Nome del file modello
model_path = 'hand_landmarker.task'

# Download del modello se non esiste
if not os.path.exists(model_path):
    print("Scaricamento del modello MediaPipe Hand Landmarker...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Modello scaricato con successo!")

# Parametri per il disegno
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 0, 0)] # Blu, Verde, Rosso, Giallo, Eraser
color_index = 0
brush_thickness = 5
eraser_thickness = 50

# Canvas per il disegno
canvas = None
prev_x, prev_y = 0, 0

# Configurazione Hand Landmarker (Nuova API)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

def main():
    global canvas, prev_x, prev_y, color_index

    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    if not success:
        print("Errore: Impossibile accedere alla webcam.")
        return
    
    h, w, c = img.shape
    canvas = np.zeros((h, w, 3), np.uint8)

    with HandLandmarker.create_from_options(options) as landmarker:
        print("Air Painter avviato con la nuova API MediaPipe! Premi 'q' per uscire.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Timestamp richiesto per la modalità VIDEO
            timestamp_ms = int(time.time() * 1000)
            
            # Esegui il rilevamento
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Interfaccia: Bottoni Colore
            cv2.rectangle(frame, (0, 0), (w, 80), (50, 50, 50), -1)
            labels = ["BLU", "VERDE", "ROSSO", "GIALLO", "CANCELLA"]
            for i, label in enumerate(labels):
                color = colors[i] if i < 4 else (200, 200, 200)
                cv2.rectangle(frame, (i * (w//5) + 5, 10), ((i+1) * (w//5) - 5, 70), color, -1)
                cv2.putText(frame, label, (i * (w//5) + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if detection_result.hand_landmarks:
                # Prendi il primo set di landmark (una sola mano)
                landmarks = detection_result.hand_landmarks[0]
                
                # Coordinate punte (8=indice, 12=medio)
                # I landmark sono normalizzati (0-1), convertiamo in pixel
                x1, y1 = int(landmarks[8].x * w), int(landmarks[8].y * h)
                x2, y2 = int(landmarks[12].x * w), int(landmarks[12].y * h)
                
                # Nocche per controllo dita alzate
                y_idx_joint = int(landmarks[6].y * h)
                y_mid_joint = int(landmarks[10].y * h)

                # Dita alzate? (punta < nocca in coordinate pixel, con y=0 in alto)
                idx_up = y1 < y_idx_joint
                mid_up = y2 < y_mid_joint

                # 1. Selezione (Indice e Medio alzati)
                if idx_up and mid_up:
                    prev_x, prev_y = 0, 0
                    if y1 < 80:
                        color_index = x1 // (w//5)
                    cv2.circle(frame, (x1, y1), 15, colors[color_index], cv2.FILLED)
                    cv2.putText(frame, "SELEZIONE", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 2. Disegno (Solo Indice alzato)
                elif idx_up and not mid_up:
                    cv2.circle(frame, (x1, y1), 10, colors[color_index], cv2.FILLED)
                    
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x1, y1
                    
                    thickness = eraser_thickness if color_index == 4 else brush_thickness
                    cv2.line(canvas, (prev_x, prev_y), (x1, y1), colors[color_index], thickness)
                    
                    prev_x, prev_y = x1, y1
                else:
                    prev_x, prev_y = 0, 0

            # Unisci Canvas e Immagine
            img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, img_inv)
            frame = cv2.bitwise_or(frame, canvas)

            cv2.imshow("Air Painter (New API)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
