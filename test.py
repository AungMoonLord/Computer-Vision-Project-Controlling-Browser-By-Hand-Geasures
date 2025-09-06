import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ตัวแปร global สำหรับควบคุม timing
prev_distance = None
last_zoom_time = 0
last_reset_time = 0
zoom_cooldown =1  # 0.5 วินาที cooldown สำหรับ zoom
reset_cooldown = 0 # 2 วินาที cooldown สำหรับ reset

def get_finger_states(landmarks):
    """ตรวจสอบสถานะของนิ้วแต่ละนิ้ว (ชูหรือไม่ชู)"""
    thumb = landmarks[4].y < landmarks[3].y
    index = landmarks[8].y < landmarks[6].y
    middle = landmarks[12].y < landmarks[10].y
    ring = landmarks[16].y < landmarks[14].y
    pinky = landmarks[20].y < landmarks[18].y
    return [thumb, index, middle, ring, pinky]

def calculate_distance(point1, point2):
    """คำนวณระยะห่างระหว่าง 2 จุด"""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calculate_zoom_gesture(thumb_tip, index_tip, landmarks):
    global prev_distance, last_zoom_time

    # ตรวจสอบว่านิ้วโป้งและนิ้วชี้ชูขึ้นหรือไม่
    thumb_up = landmarks[4].y < landmarks[3].y
    index_up = landmarks[8].y < landmarks[6].y

    # ถ้าไม่ได้ชูทั้งสองนิ้ว ให้หยุดทำงาน
    if not (thumb_up and index_up):
        return None

    current_time = time.time()
    distance = calculate_distance(thumb_tip, index_tip)

    if (current_time - last_zoom_time) < zoom_cooldown:
        return None

    if prev_distance is not None:
        diff = distance - prev_distance
        if diff > 0.03:
            pyautogui.hotkey('ctrl', '+')
            print("Zoom In")
            last_zoom_time = current_time
            return "Zoom In"
        elif diff < -0.03:
            pyautogui.hotkey('ctrl', '-')
            print("Zoom Out")
            last_zoom_time = current_time
            return "Zoom Out"

    prev_distance = distance
    return None

def is_all_fingers_up(landmarks):
    """ตรวจสอบว่าชูนิ้ว 5 นิ้วพร้อมกันหรือไม่"""
    thumb = landmarks[4].y < landmarks[3].y
    index = landmarks[8].y < landmarks[6].y
    middle = landmarks[12].y < landmarks[10].y
    ring = landmarks[16].y < landmarks[14].y
    pinky = landmarks[20].y < landmarks[18].y
    
    # ตรวจสอบว่าทุกนิ้วชูขึ้น
    return thumb and index and middle and ring and pinky

# เปิดกล้อง
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # แปลงภาพสำหรับ MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ตรวจจับมือ
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # วาดจุด landmark
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ดึงข้อมูล landmark
                landmarks = hand_landmarks.landmark
                
                # ควบคุม Zoom ด้วยนิ้วโป้ง + นิ้วชี้
                thumb_tip = landmarks[4]   # นิ้วโป้ง
                index_tip = landmarks[8]   # นิ้วชี้
                zoom_action = calculate_zoom_gesture(thumb_tip, index_tip, landmarks)
                
                # ตรวจสอบชูนิ้ว 5 นิ้ว (Reset Zoom) พร้อม cooldown
                current_time = time.time()
                if is_all_fingers_up(landmarks) and (current_time - last_reset_time) > reset_cooldown:
                    pyautogui.hotkey('ctrl', '0')
                    print("Reset Zoom")
                    last_reset_time = current_time
                    cv2.putText(image, "Reset Zoom", (50, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # แสดงค่าระยะห่าง
                distance = calculate_distance(thumb_tip, index_tip)
                cv2.putText(image, f"Distance: {distance:.2f}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # แสดง action ที่ทำ
                if zoom_action:
                    cv2.putText(image, zoom_action, (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # แสดงผล
        cv2.imshow('Zoom Gesture Control', image)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
