import cv2
import mediapipe as mp
import pyautogui
import time

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_action_time = 0
cooldown = 0.5  # หน่วงเวลา 0.5 วินาที

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # กลับด้านเหมือนกระจก
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        action = "NONE"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[4]  # ปลายนิ้วโป้ง
                index_tip = hand_landmarks.landmark[8]  # ปลายนิ้วชี้

                # ตรวจสอบว่ามือกำหรือไม่ (fist)
                fist = abs(thumb_tip.y - index_tip.y) < 0.05 and abs(thumb_tip.x - index_tip.x) < 0.1
                if fist:
                    action = "STOP"
                else:
                    now = time.time()

                    # กำหนดความคลาดเคลื่อนของตำแหน่ง
                    y_threshold = 0.05  # ถ้า y ต่างกันน้อยกว่านี้ ถือว่าอยู่ตรงกลาง
                    x_threshold = 0.1   # ถ้า x ต่างกันน้อยกว่านี้ ถือว่าอยู่ตรงกลาง

                    # ตรวจสอบว่า นิ้วโป้งอยู่บนหรือล่าง
                    if thumb_tip.y < index_tip.y - y_threshold:  # นิ้วโป้งอยู่บน
                        if now - last_action_time > cooldown:
                            pyautogui.press("left")
                            action = "LEFT"
                            last_action_time = now
                    elif thumb_tip.y > index_tip.y + y_threshold:  # นิ้วโป้งอยู่ล่าง
                        if now - last_action_time > cooldown:
                            pyautogui.press("right")
                            action = "RIGHT"
                            last_action_time = now
                    else:
                        # นิ้วโป้งอยู่ตรงกลาง (ไม่ส่งสัญญาณ)
                        action = "NONE"

        # แสดง Action บนหน้าจอ
        cv2.putText(frame, action, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Hand Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # กด ESC เพื่อออก
            break

cap.release()
cv2.destroyAllWindows()