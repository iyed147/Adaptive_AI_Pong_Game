import cv2
import mediapipe as mp
import pandas as pd
import os
import time
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)

LABEL_KEYS = {"1": "UP", "2": "DOWN", "3": "NONE"}

TARGET_PER_LABEL = 1000   # 1000 x 3 = 3000
CAPTURE_DELAY = 0.03      # ~30 FPS (très rapide)

# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

# =========================================================
# DATASET
# =========================================================

data = []
last_capture = 0
current_label = None

count = {"UP": 0, "DOWN": 0, "NONE": 0}

print("\n=== AUTO DATASET COLLECT ===")
print("1 -> UP | 2 -> DOWN | 3 -> NONE")
print("q -> QUIT\n")

# =========================================================
# LOOP
# =========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        pressed = chr(key)
        if pressed in LABEL_KEYS:
            current_label = LABEL_KEYS[pressed]
            print(f"Label actif: {current_label}")
        elif pressed == "q":
            break

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y

            features = []
            for lm in hand_landmarks.landmark:
                features.append(lm.x - wrist_x)
                features.append(lm.y - wrist_y)
                features.append(lm.z)

            current_time = time.time()

            if (
                current_label
                and current_time - last_capture > CAPTURE_DELAY
                and count[current_label] < TARGET_PER_LABEL
            ):
                data.append(features + [current_label])
                count[current_label] += 1
                last_capture = current_time

                print(
                    f"{current_label}: {count[current_label]}/{TARGET_PER_LABEL} | "
                    f"Total: {len(data)}"
                )

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Affichage
    cv2.putText(frame, f"Label: {current_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"UP: {count['UP']}  DOWN: {count['DOWN']}  NONE: {count['NONE']}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Collect Gesture Dataset", frame)

    # Fin automatique si complet
    if all(count[l] >= TARGET_PER_LABEL for l in count):
        print("\n✅ Tous les samples collectés.")
        break

# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()
hands.close()

# =========================================================
# SAVE
# =========================================================

if data:
    columns = []
    for i in range(21):
        columns.append(f"x{i}")
        columns.append(f"y{i}")
        columns.append(f"z{i}")
    columns.append("label")

    df = pd.DataFrame(data, columns=columns)

    filename = f"gesture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_path = os.path.join(DATA_DIR, filename)

    df.to_csv(save_path, index=False)

    print("\n✅ Dataset sauvegardé :")
    print(save_path)
    print(f"Total samples : {len(df)}")
else:
    print("\nAucune donnée enregistrée.")