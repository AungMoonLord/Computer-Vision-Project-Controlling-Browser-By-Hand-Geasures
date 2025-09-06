import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ตัวแปร global
prev_distance = None
last_reset_time = 0
reset_cooldown = 2.0  # 2 วินาที cooldown

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

def calculate_zoom_gesture(thumb_tip, index_tip):
    """ควบคุม Zoom In/Out ด้วยการกาง/หุบนิ้วโป้งกับนิ้วชี้"""
    global prev_distance
    
    distance = calculate_distance(thumb_tip, index_tip)
    
    if prev_distance is not None:
        diff = distance - prev_distance
        if diff > 0.02:  # กางออก
            pyautogui.hotkey('ctrl', '+')
            print("Zoom In")
            return "Zoom In"
        elif diff < -0.02:  # หุบเข้า
            pyautogui.hotkey('ctrl', '-')
            print("Zoom Out")
            return "Zoom Out"
    
    prev_distance = distance
    return None

def is_fist(landmarks):
    """ตรวจสอบว่ากำมือหรือไม่ด้วยเงื่อนไขที่เข้มงวดขึ้น"""
    # ตรวจสอบว่าปลายนิ้วทุกนิ้วอยู่ต่ำกว่า MCP joints
    thumb_tip = landmarks[4].y > landmarks[3].y  # นิ้วโป้ง
    index_tip = landmarks[8].y > landmarks[6].y  # นิ้วชี้
    middle_tip = landmarks[12].y > landmarks[10].y  # นิ้วกลาง
    ring_tip = landmarks[16].y > landmarks[14].y  # นิ้วนาง
    pinky_tip = landmarks[20].y > landmarks[18].y  # นิ้วก้อย
    
    # ตรวจสอบว่าทุกนิ้วหุบ
    all_fingers_folded = thumb_tip and index_tip and middle_tip and ring_tip and pinky_tip
    
    # เพิ่มเงื่อนไขระยะห่างระหว่างนิ้วเพื่อความแม่นยำ
    if all_fingers_folded:
        # ตรวจสอบว่าปลายนิ้วชี้กับนิ้วกลางใกล้กัน (กำมือแน่น)
        index_middle_distance = calculate_distance(landmarks[8], landmarks[12])
        return index_middle_distance < 0.05  # ค่า threshold สำหรับกำมือ
    
    return False

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
                zoom_action = calculate_zoom_gesture(thumb_tip, index_tip)
                
                # ตรวจสอบกำมือ (Reset Zoom) พร้อม cooldown
                current_time = time.time()
                if is_fist(landmarks) and (current_time - last_reset_time) > reset_cooldown:
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
