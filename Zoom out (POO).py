import cv2
import mediapipe as mp
import pyautogui
import time

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# ตัวแปรควบคุมการกดปุ่ม
last_zoom_time = 0
delay = 2  # หน่วงเวลา 2 วินาที

def is_victory_sign(hand_landmarks):
    """ตรวจสอบว่าเป็นสัญลักษณ์ V (แบมือ) หรือไม่"""
    landmarks = hand_landmarks.landmark

    # ตรวจสอบว่านิ้วชี้และนิ้วกลางเหยียดตรง แต่นิ้วนางและนิ้วก้อยงอ
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # ตรวจสอบว่านิ้วชี้และนิ้วกลางอยู่เหนือนิ้วก้อย (งอ)
    if (index_tip.y < ring_tip.y and
        middle_tip.y < ring_tip.y and
        index_tip.y < pinky_tip.y and
        middle_tip.y < pinky_tip.y and
        thumb_tip.y > index_tip.y):  # หัวแม่มืออยู่ต่ำกว่า index
        return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip และแปลงภาพเป็น RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ตรวจสอบสัญลักษณ์ V
            if is_victory_sign(hand_landmarks):
                current_time = time.time()
                if current_time - last_zoom_time > delay:
                    pyautogui.hotkey('ctrl', '-')
                    print("Zoom In!")
                    last_zoom_time = current_time

    cv2.imshow('Hand Gesture Zoom Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()