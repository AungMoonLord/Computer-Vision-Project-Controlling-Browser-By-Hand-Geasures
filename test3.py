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
last_zoom_time = 0
last_reset_time = 0
zoom_cooldown = 0.3   # ⚡ ลด cooldown ให้ trigger ไวขึ้น
reset_cooldown = 2.0

# ✅ ตั้งค่า threshold แบบยืดหยุ่น — คุณจะได้ปรับเองจากค่าจริง
ZOOM_IN_THRESH = 0.20
ZOOM_OUT_THRESH = 0.08  # เริ่มที่ 0.08 — แล้วกด 'c' เพื่อ calibrate

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def is_thumb_and_index_up(landmarks):
    """✅ แก้: ผ่อนปรนเงื่อนไข — ให้ Zoom ได้แม้นิ้วอื่นจะยกเล็กน้อย"""
    thumb_up = landmarks[4].y < landmarks[3].y
    index_up = landmarks[8].y < landmarks[6].y
    # ไม่ต้องเช็กนิ้วอื่น — ปล่อยให้ผู้ใช้จีบมือตามธรรมชาติ
    return thumb_up and index_up

def is_all_fingers_up(landmarks):
    thumb = landmarks[4].y < landmarks[3].y
    index = landmarks[8].y < landmarks[6].y
    middle = landmarks[12].y < landmarks[10].y
    ring = landmarks[16].y < landmarks[14].y
    pinky = landmarks[20].y < landmarks[18].y
    return thumb and index and middle and ring and pinky

def calculate_zoom_gesture(thumb_tip, index_tip):
    global prev_distance, last_zoom_time, ZOOM_OUT_THRESH
    
    current_time = time.time()
    distance = calculate_distance(thumb_tip, index_tip)
    
    if (current_time - last_zoom_time) < zoom_cooldown:
        return None

    # 🔍 Debug: แสดงค่าระยะใน console ทุกเฟรม (ขณะ gesture ถูกต้อง)
    print(f"DEBUG → Distance: {distance:.4f} | Zoom Out Threshold: {ZOOM_OUT_THRESH}")

    if distance > ZOOM_IN_THRESH:
        pyautogui.hotkey('ctrl', '+')
        print(f"🟢 [ZOOM IN] Distance: {distance:.4f}")
        last_zoom_time = current_time
        prev_distance = distance
        return "Zoom In"
    elif distance < ZOOM_OUT_THRESH:
        pyautogui.hotkey('ctrl', '-')
        print(f"🔴 [ZOOM OUT] Distance: {distance:.4f} ✅ TRIGGERED!")
        last_zoom_time = current_time
        prev_distance = distance
        return "Zoom Out"

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

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                landmarks = hand_landmarks.landmark

                # ✅ ตรวจจับ gesture ถ้าแค่นิ้วโป้ง+ชี้ยก → ไม่สนใจนิ้วอื่น
                if is_thumb_and_index_up(landmarks):
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    zoom_action = calculate_zoom_gesture(thumb_tip, index_tip)
                    
                    distance = calculate_distance(thumb_tip, index_tip)
                    # แสดงค่า Distance บนหน้าจอ — ดูแบบเรียลไทม์
                    cv2.putText(image, f"Dist: {distance:.4f}", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    if zoom_action:
                        color = (0, 255, 255) if "In" in zoom_action else (0, 0, 255)
                        cv2.putText(image, zoom_action, (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # ✅ แสดงสถานะว่า gesture ตรวจจับได้หรือไม่
                    cv2.putText(image, "Gesture: ACTIVE", (50, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Reset Zoom
                current_time = time.time()
                if is_all_fingers_up(landmarks) and (current_time - last_reset_time) > reset_cooldown:
                    pyautogui.hotkey('ctrl', '0')
                    print("[🔄 RESET ZOOM]")
                    last_reset_time = current_time
                    cv2.putText(image, "Reset Zoom", (50, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # ✅ กด 'c' เพื่อ Calibrate Zoom Out Threshold
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    if is_thumb_and_index_up(landmarks):
                        dist = calculate_distance(landmarks[4], landmarks[8])
                        ZOOM_OUT_THRESH = dist + 0.01  # บวกเผื่อไว้นิดหน่อย
                        print(f"🎯 AUTO-CALIBRATED! New ZOOM_OUT_THRESH = {ZOOM_OUT_THRESH:.4f}")

        cv2.imshow('Hand Gesture Browser Zoom Control', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()