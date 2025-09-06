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
zoom_cooldown = 0.8   # เวลาหน่วงระหว่างการ zoom
reset_cooldown = 2.0  # เวลาหน่วงหลัง reset zoom

def calculate_distance(point1, point2):
    """คำนวณระยะห่างระหว่าง 2 จุด landmark"""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def is_thumb_and_index_up(landmarks):
    """ตรวจสอบว่านิ้วโป้งและนิ้วชี้ยกขึ้น และนิ้วอื่นไม่ยกขึ้น"""
    thumb_up = landmarks[4].y < landmarks[3].y  # นิ้วโป้งยกขึ้น
    index_up = landmarks[8].y < landmarks[6].y  # นิ้วชี้ยกขึ้น
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    
    # นิ้วกลาง, นิ้วนาง, นิ้วก้อย ไม่ควรยก
    other_fingers_up = middle_up or ring_up or pinky_up
    
    return thumb_up and index_up and not other_fingers_up

def is_all_fingers_up(landmarks):
    """ตรวจสอบว่าชูนิ้ว 5 นิ้วพร้อมกันหรือไม่"""
    thumb = landmarks[4].y < landmarks[3].y
    index = landmarks[8].y < landmarks[6].y
    middle = landmarks[12].y < landmarks[10].y
    ring = landmarks[16].y < landmarks[14].y
    pinky = landmarks[20].y < landmarks[18].y
    
    return thumb and index and middle and ring and pinky

def calculate_zoom_gesture(thumb_tip, index_tip):
    """ควบคุม Zoom In/Out ด้วยระยะห่างระหว่างนิ้วโป้งและนิ้วชี้"""
    global prev_distance, last_zoom_time
    
    current_time = time.time()
    distance = calculate_distance(thumb_tip, index_tip)
    
    # ตรวจสอบ cooldown
    if (current_time - last_zoom_time) < zoom_cooldown:
        return None

    # กำหนดค่า threshold สำหรับการซูม (ปรับตามความเหมาะสม)
    ZOOM_IN_THRESH = 0.20
    ZOOM_OUT_THRESH = 0.03  # เริ่มที่ 0.12 — คุณสามารถลดลงเป็น 0.08 ได้ถ้าจีบมือแล้วค่าต่ำพอ

    if distance > ZOOM_IN_THRESH:
        pyautogui.hotkey('ctrl', '+')
        print(f"[ZOOM IN] Distance: {distance:.3f}")
        last_zoom_time = current_time
        prev_distance = distance
        return "Zoom In"
    elif distance < ZOOM_OUT_THRESH:
        pyautogui.hotkey('ctrl', '-')
        print(f"[ZOOM OUT] Distance: {distance:.3f}")
        last_zoom_time = current_time
        prev_distance = distance
        return "Zoom Out"

    # อัปเดตค่าระยะล่าสุด แม้ยังไม่ trigger
    prev_distance = distance
    return None

# เปิดกล้อง
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # แปลงภาพสำหรับ MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ตรวจจับมือ
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # วาด landmark
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                landmarks = hand_landmarks.landmark

                # ควบคุม Zoom ด้วยนิ้วโป้ง + นิ้วชี้
                if is_thumb_and_index_up(landmarks):
                    thumb_tip = landmarks[4]   # ปลายนิ้วโป้ง
                    index_tip = landmarks[8]   # ปลายนิ้วชี้
                    zoom_action = calculate_zoom_gesture(thumb_tip, index_tip)
                    
                    # แสดงระยะห่างบนหน้าจอ
                    distance = calculate_distance(thumb_tip, index_tip)
                    cv2.putText(image, f"Distance: {distance:.3f}", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # แสดง action ที่ trigger
                    if zoom_action:
                        color = (0, 255, 255) if "In" in zoom_action else (255, 0, 255)
                        cv2.putText(image, zoom_action, (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Reset Zoom ด้วยการชู 5 นิ้ว
                current_time = time.time()
                if is_all_fingers_up(landmarks) and (current_time - last_reset_time) > reset_cooldown:
                    pyautogui.hotkey('ctrl', '0')
                    print("[RESET ZOOM]")
                    last_reset_time = current_time
                    cv2.putText(image, "Reset Zoom", (50, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # แสดงผล
        cv2.imshow('Hand Gesture Browser Zoom Control', image)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # กด 'c' เพื่อ calibrate — ดูค่าระยะตอนจีบมือ
        if cv2.waitKey(10) & 0xFF == ord('c'):
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    if is_thumb_and_index_up(landmarks):
                        thumb_tip = landmarks[4]
                        index_tip = landmarks[8]
                        dist = calculate_distance(thumb_tip, index_tip)
                        print(f"📌 Calibrated Min Distance: {dist:.3f} — ลองใช้ค่านี้เป็น ZOOM_OUT_THRESH")

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
