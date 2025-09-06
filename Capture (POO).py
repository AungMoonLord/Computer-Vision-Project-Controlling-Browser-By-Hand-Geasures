import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# เปิดกล้อง
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

# ตัวแปรควบคุม
last_screenshot_time = 0
delay = 2  # หน่วงเวลา 2 วินาที

def is_three_fingers_up(hand_landmarks):
    """ตรวจสอบว่า 3 นิ้วแรกชี้ขึ้น (นิ้วชี้, นิ้วกลาง, นิ้วนาง)"""
    landmarks = hand_landmarks.landmark
    
    try:
        # ตรวจสอบ 3 นิ้วแรกชี้ขึ้น
        index_up = landmarks[8].y < landmarks[6].y   # นิ้วชี้
        middle_up = landmarks[12].y < landmarks[10].y  # นิ้วกลาง  
        ring_up = landmarks[16].y < landmarks[14].y    # นิ้วนาง
        
        # ตรวจสอบนิ้วก้อยงอ
        pinky_down = landmarks[20].y > landmarks[18].y
        
        # ตรวจสอบว่านิ้วที่ชี้ขึ้นมีการแยกออกจากกัน
        index_middle_separated = abs(landmarks[8].x - landmarks[12].x) > 0.03
        middle_ring_separated = abs(landmarks[12].x - landmarks[16].x) > 0.03
        
        result = (index_up and middle_up and ring_up and pinky_down and 
                 index_middle_separated and middle_ring_separated)
        
        return result
    except:
        return False

def is_hand_closed(hand_landmarks):
    """ตรวจสอบว่ามือหุบ"""
    try:
        landmarks = hand_landmarks.landmark
        # ตรวจสอบว่านิ้วชี้งอ
        index_folded = landmarks[8].y > landmarks[6].y
        return index_folded
    except:
        return True

# ปิด fail-safe
pyautogui.FAILSAFE = False

screenshot_count = 0

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

            # ตรวจสอบว่ามือไม่ได้หุบ และ 3 นิ้วชี้ขึ้น
            if not is_hand_closed(hand_landmarks) and is_three_fingers_up(hand_landmarks):
                current_time = time.time()
                if current_time - last_screenshot_time > delay:
                    try:
                        # ถ่าย screenshot
                        screenshot = pyautogui.screenshot()
                        screenshot_count += 1
                        
                        # บันทึกไฟล์
                        filename = f"screenshot_{screenshot_count}_{int(time.time())}.png"
                        screenshot.save(filename)
                        
                        print(f"Screenshot saved: {filename}")
                        last_screenshot_time = current_time
                        
                        # แสดงผลบนหน้าจอ
                        cv2.putText(frame, "SCREENSHOT TAKEN!", (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                    except Exception as e:
                        print(f"Error taking screenshot: {e}")

    # แสดงคำแนะนำ
    cv2.putText(frame, "Show 3 fingers up to take screenshot", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Screenshots: {screenshot_count}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('3 Finger Screenshot Capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
