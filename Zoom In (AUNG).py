import cv2
import mediapipe as mp
import pyautogui
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ตัวแปร global
prev_distance = None

def calculate_zoom_gesture(thumb_tip, index_tip):
    global prev_distance
    
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x)**2 +
        (thumb_tip.y - index_tip.y)**2
    )
    
    if prev_distance is not None:
        diff = distance - prev_distance
        if diff > 0.02:
            pyautogui.hotkey('ctrl', '+')
            print("Zoom In")
            return "Zoom In"
        elif diff < -0.02:
            pyautogui.hotkey('ctrl', '-')
            print("Zoom Out")
            return "Zoom Out"
    
    prev_distance = distance
    return None

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                
                action = calculate_zoom_gesture(thumb_tip, index_tip)
                
                distance = math.sqrt(
                    (thumb_tip.x - index_tip.x)**2 +
                    (thumb_tip.y - index_tip.y)**2
                )
                
                cv2.putText(image, f"Dist: {distance:.2f}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if action:
                    cv2.putText(image, action, (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Zoom Gesture Control', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
