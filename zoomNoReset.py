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
zoom_cooldown = 0.5  # 0.5 วินาที cooldown สำหรับ zoom

def calculate_distance(point1, point2):
    """คำนวณระยะห่างระหว่าง 2 จุด"""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def is_thumb_and_index_up(landmarks):
    """ตรวจสอบว่านิ้วโป้งและนิ้วชี้ยกขึ้น และนิ้วอื่นไม่ยกขึ้น"""
    thumb_up = landmarks[4].y < landmarks[3].y  # นิ้วโป้งยกขึ้น
    index_up = landmarks[8].y < landmarks[6].y  # นิ้วชี้ยกขึ้น
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    
    # นิ้วอื่นยกขึ้นไหม?
    other_fingers_up = middle_up or ring_up or pinky_up
    
    return thumb_up and index_up and not other_fingers_up

def calculate_zoom_gesture(thumb_tip, index_tip):
    """ควบคุม Zoom In/Out ด้วยระยะห่างระหว่างนิ้วโป้งและนิ้วชี้"""
    global prev_distance, last_zoom_time
    
    current_time = time.time()
    distance = calculate_distance(thumb_tip, index_tip)
    
    # ตรวจสอบ cooldown
    if (current_time - last_zoom_time) < zoom_cooldown:
        return None
    
    # กำหนด threshold สำหรับ "ห่างกันมาก" / "ห่างกันน้อย"
    threshold = 0.15  # ค่า normalized (0-1) อาจต้องปรับตามกล้อง
    
    if prev_distance is not None:
        # ถ้าระยะห่างมากกว่า threshold → Zoom In
        if distance > threshold:
            pyautogui.hotkey('ctrl', '+')
            print("Zoom In")
            last_zoom_time = current_time
            return "Zoom In"
        # ถ้าระยะห่างน้อยกว่า threshold → Zoom Out
        elif distance < threshold:
            pyautogui.hotkey('ctrl', '-')
            print("Zoom Out")
            last_zoom_time = current_time
            return "Zoom Out"
    
    prev_distance = distance
    return None

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
                
                # ตรวจสอบว่านิ้วโป้งและนิ้วชี้ยกขึ้นไหม
                if is_thumb_and_index_up(landmarks):
                    # ดึงตำแหน่งปลายนิ้ว
                    thumb_tip = landmarks[4]   # นิ้วโป้ง
                    index_tip = landmarks[8]   # นิ้วชี้
                    
                    # คำนวณการซูม
                    zoom_action = calculate_zoom_gesture(thumb_tip, index_tip)
                    
                    # แสดงค่าระยะห่าง
                    distance = calculate_distance(thumb_tip, index_tip)
                    cv2.putText(image, f"Distance: {distance:.2f}", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # แสดง action ที่ทำ
                    if zoom_action:
                        cv2.putText(image, zoom_action, (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    # ถ้านิ้วโป้งหรือนิ้วชี้ไม่ยกขึ้น → ไม่ทำอะไร
                    pass

        # แสดงผล
        cv2.imshow('Zoom Gesture Control', image)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()