import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import urllib.request

class HandTracker:
    def __init__(self, model_path='hand_landmarker.task'):
        self.model_path = model_path
        self._check_model()
        
        # Configurazione della nuova API Tasks
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def _check_model(self):
        if not os.path.exists(self.model_path):
            print(f"Scaricamento del modello in corso...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

    def find_hands(self, frame):
        # Converte il frame per MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.time() * 1000)
        self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        return self.results

    def get_landmarks_pixel(self, frame_shape):
        if not self.results or not self.results.hand_landmarks:
            return None
        
        h, w, _ = frame_shape
        landmarks = []
        for lm in self.results.hand_landmarks[0]:
            landmarks.append({'x': int(lm.x * w), 'y': int(lm.y * h), 'z': lm.z})
        return landmarks

    def fingers_up(self, landmarks):
        """
        Ritorna una lista di 5 valori (1 se il dito è su, 0 se è giù).
        Ordine: Pollice, Indice, Medio, Anulare, Mignolo.
        """
        if not landmarks:
            return [0, 0, 0, 0, 0]
        
        fingers = []
        
        # Pollice (confronto x perché si muove lateralmente)
        # Nota: assumiamo mano destra con palmo verso la camera o invertito dal flip
        if landmarks[4]['x'] < landmarks[3]['x']:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Dita (confronto y: punta < nocca)
        tip_ids = [8, 12, 16, 20]
        joint_ids = [6, 10, 14, 18]
        
        for tip, joint in zip(tip_ids, joint_ids):
            if landmarks[tip]['y'] < landmarks[joint]['y']:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def close(self):
        self.detector.close()
