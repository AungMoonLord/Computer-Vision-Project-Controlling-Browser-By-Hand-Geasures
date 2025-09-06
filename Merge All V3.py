# Import library ทั้งหมด
import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

# ตั้งค่าเริ่มต้น
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# เปิดกล้องเพียงครั้งเดียว
cap = cv2.VideoCapture(0)

# ตัวแปรควบคุม global (รวมจากทุกไฟล์)
last_scroll_time = 0
last_screenshot_time = 0
last_zoom_time = 0
last_reset_time = 0
# ...และตัวแปรอื่นๆ

def is_thumb_and_index_up(landmarks):
    # โค้ดจาก inout_ver3.py
    ...

def is_all_fingers_up(landmarks):
    # โค้ดจาก inout_ver3.py
    ...

def is_three_fingers_up(landmarks):
    # โค้ดจาก Capture (POO).py
    ...
    
def calculate_zoom_gesture(thumb_tip, index_tip):
    # โค้ดจาก inout_ver3.py
    ...

# ...สร้างฟังก์ชันสำหรับท่าทางอื่นๆ ที่จำเป็น

# ลูปหลัก
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    action = "NONE"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # ใช้ if/elif เพื่อกำหนดลำดับความสำคัญของท่าทาง
            # ตรวจจับท่า Zoom/Reset ก่อน
            if is_all_fingers_up(landmarks):
                 # โค้ด Reset Zoom จาก inout_ver3.py
                 ...
            elif is_thumb_and_index_up(landmarks):
                # โค้ด Zoom In/Out จาก inout_ver3.py
                ...
            # ตรวจจับท่า Screenshot
            elif is_three_fingers_up(hand_landmarks):
                # โค้ด Screenshot จาก Capture (POO).py
                ...
            # ตรวจจับท่า Scroll Up/Down
            #elif ...:
                # โค้ด Scroll Up/Down จาก Scroll up - Scroll down (BAM).py
                ...
            # ตรวจจับท่า Scroll Left/Right
            #elif ...:
                # โค้ด Scroll Left/Right จาก Scroll Left - Scroll Right (WHENG).py
                ...
            else:
                action = "NONE"
    
    cv2.putText(frame, action, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Combined Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
hands.close()