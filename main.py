import cv2
import numpy as np
from hand_tracker import HandTracker

def main():
    # Inizializza tracciatore mano
    tracker = HandTracker()
    
    # Inizializza webcam
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    if not success:
        print("Errore: Impossibile accedere alla webcam.")
        return

    h, w, c = img.shape
    canvas = np.zeros((h, w, 3), np.uint8)

    # Colori (BGR)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 0, 0)] # Blu, Verde, Rosso, Giallo, Eraser
    color_index = 0
    brush_thickness = 10
    eraser_thickness = 80
    prev_x, prev_y = 0, 0

    # Definiamo le connessioni dello scheletro della mano (coppie di landmark)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),    # Pollice
        (0, 5), (5, 6), (6, 7), (7, 8),    # Indice
        (5, 9), (9, 10), (10, 11), (11, 12), # Medio
        (9, 13), (13, 14), (14, 15), (15, 16), # Anulare
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Mignolo e base
    ]

    print("Air Painter 2.0 Avviato! Usa 5 dita per la Gomma Totale.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        
        # 1. Trova le mani
        tracker.find_hands(frame)
        landmarks = tracker.get_landmarks_pixel(frame.shape)

        # 2. Interfaccia Colori (Header)
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
        labels = ["BLU", "VERDE", "ROSSO", "GIALLO", "GOMMA MANUALE"]
        for i, label in enumerate(labels):
            color = colors[i] if i < 4 else (150, 150, 150)
            cv2.rectangle(frame, (i * (w//5) + 5, 10), ((i+1) * (w//5) - 5, 70), color, -1)
            cv2.putText(frame, label, (i * (w//5) + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if landmarks:
            # Disegna lo scheletro (come nella foto)
            for connection in HAND_CONNECTIONS:
                p1 = (landmarks[connection[0]]['x'], landmarks[connection[0]]['y'])
                p2 = (landmarks[connection[1]]['x'], landmarks[connection[1]]['y'])
                cv2.line(frame, p1, p2, (0, 0, 255), 2) # Linee rosse
            
            for lm in landmarks:
                cv2.circle(frame, (lm['x'], lm['y']), 4, (255, 255, 255), -1) # Punti bianchi

            # Ottieni stato dita
            fingers = tracker.fingers_up(landmarks)
            
            x1, y1 = landmarks[8]['x'], landmarks[8]['y'] # Punta Indice
            x2, y2 = landmarks[12]['x'], landmarks[12]['y'] # Punta Medio

            # GESTO 1: GOMMA TOTALE (Mano Aperta)
            if sum(fingers) == 5:
                # Se la mano è tutta aperta, disegna un cerchio grande di "cancellazione" sul canvas
                cv2.circle(frame, (x1, y1), eraser_thickness, (255, 255, 255), 2)
                cv2.putText(frame, "CANCELLAZIONE PALMO", (x1-50, y1-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(canvas, (x1, y1), eraser_thickness, (0,0,0), -1)
                prev_x, prev_y = 0, 0

            # GESTO 2: SELEZIONE (Indice e Medio su)
            elif fingers[1] and fingers[2]:
                prev_x, prev_y = 0, 0
                if y1 < 80:
                    color_index = x1 // (w//5)
                cv2.rectangle(frame, (x1-15, y1-15), (x1+15, y1+15), colors[color_index], -1)
                cv2.putText(frame, "MODALITA SELEZIONE", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # GESTO 3: DISEGNO (Solo Indice su)
            elif fingers[1] and not fingers[2]:
                cv2.circle(frame, (x1, y1), 10, colors[color_index], cv2.FILLED)
                
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1
                
                thickness = eraser_thickness if color_index == 4 else brush_thickness
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), colors[color_index], thickness)
                
                prev_x, prev_y = x1, y1
            else:
                prev_x, prev_y = 0, 0

        # Unisci Canvas e Immagine Video
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, canvas)

        cv2.imshow("Air Painter Pro", frame)
        
        # Premi 'c' per pulire tutto istantaneamente
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((h, w, 3), np.uint8)

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
