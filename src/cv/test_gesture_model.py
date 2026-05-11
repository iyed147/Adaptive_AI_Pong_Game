import cv2
import mediapipe as mp
import joblib
import os
import numpy as np

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "gesture_model.pkl")

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)
print("✅ Modèle chargé :", MODEL_PATH)

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
# LOOP
# =========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = "NONE"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y

            features = []
            for lm in hand_landmarks.landmark:
                features.append(lm.x - wrist_x)
                features.append(lm.y - wrist_y)
                features.append(lm.z)

            X = np.array(features).reshape(1, -1)
            label = model.predict(X)[0]

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(frame, f"Pred: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Test Gesture Model", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()