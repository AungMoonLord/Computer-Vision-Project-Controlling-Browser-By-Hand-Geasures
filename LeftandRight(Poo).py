import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05

cap = cv2.VideoCapture(0)

last_action_time = 0
cooldown = 0.2

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        action = "NONE"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ดึงตำแหน่ง landmark
                landmarks = hand_landmarks.landmark
                
                # ปลายนิ้วชี้และนิ้วกลางเท่านั้น
                index_tip = landmarks[8]   # นิ้วชี้
                middle_tip = landmarks[12] # นิ้วกลาง

                now = time.time()

                # โฟกัสเฉพาะที่นิ้วชี้กับนิ้วกลาง
                # คำนวณระยะห่างในแนวแนวนอน
                finger_x_distance = abs(index_tip.x - middle_tip.x)
                
                # คำนวณระยะห่างในแนวแนวตั้ง
                finger_y_distance = abs(index_tip.y - middle_tip.y)
                
                # ตรวจสอบว่านิ้วทั้งสองขึ้น (ไม่งอ)
                index_up = index_tip.y < landmarks[5].y  # โคนนิ้วชี้
                middle_up = middle_tip.y < landmarks[9].y  # โคนนิ้วกลาง
                
                fingers_up = index_up and middle_up

                # แสดงค่าเพื่อ debug
                print(f"X Distance: {finger_x_distance:.3f}, Y Distance: {finger_y_distance:.3f}")

                # ซ้าย: นิ้วชี้กับกลางขึ้น + ชิดกัน
                if fingers_up and finger_x_distance < 0.04:
                    if now - last_action_time > cooldown:
                        try:
                            pyautogui.press('left')
                            print("LEFT - Fingers close together")
                        except Exception as e:
                            print(f"Left arrow failed: {e}")
                        
                        action = "LEFT"
                        last_action_time = now

                # ขวา: นิ้วชี้กับกลางขึ้น + แยกกัน
                elif fingers_up and finger_x_distance > 0.07:
                    if now - last_action_time > cooldown:
                        try:
                            pyautogui.press('right')
                            print("RIGHT - Fingers apart")
                        except Exception as e:
                            print(f"Right arrow failed: {e}")
                        
                        action = "RIGHT"
                        last_action_time = now

                # พร้อมใช้งาน
                elif fingers_up:
                    action = "READY"

        # แสดง Action บนหน้าจอ
        cv2.putText(frame, action, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # แสดงคำแนะนำ
        cv2.putText(frame, "Left: Index & middle close", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Right: Index & middle apart", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Finger Focus Only", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()