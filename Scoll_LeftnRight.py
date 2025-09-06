import cv2
import mediapipe as mp
import pyautogui
import time

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# ตัวแปรควบคุม
last_scroll_time = 0
scroll_cooldown = 0.02  # เพิ่มเวลาหน่วงเพื่อความเสถียร
distance_threshold = 0.05  # ระยะห่างในหน่วย normalized (0-1)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # กลับด้านภาพเพื่อให้เหมือนกระจก
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        action = "NONE"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # วาดจุด landmark บนมือ
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ตำแหน่งปลายนิ้ว
                index_tip = hand_landmarks.landmark[8]     # ปลายนิ้วชี้
                middle_tip = hand_landmarks.landmark[12]   # ปลายนิ้วกลาง

                # ตรวจสอบว่านิ้วชี้และนิ้วกลางยกขึ้นไหม
                def is_finger_up(tip, mcp):
                    return tip.y < mcp.y

                index_mcp = hand_landmarks.landmark[5]
                middle_mcp = hand_landmarks.landmark[9]

                index_up = is_finger_up(index_tip, index_mcp)
                middle_up = is_finger_up(middle_tip, middle_mcp)

                # ตรวจสอบว่านิ้วโป้งยกขึ้นไหม
                thumb_tip = hand_landmarks.landmark[4]
                thumb_mcp = hand_landmarks.landmark[1]
                thumb_up = thumb_tip.x < thumb_mcp.x  # นิ้วโป้งหงาย

                # นิ้วอื่นยกขึ้นไหม?
                ring_tip = hand_landmarks.landmark[16]
                ring_mcp = hand_landmarks.landmark[13]
                ring_up = is_finger_up(ring_tip, ring_mcp)

                pinky_tip = hand_landmarks.landmark[20]
                pinky_mcp = hand_landmarks.landmark[17]
                pinky_up = is_finger_up(pinky_tip, pinky_mcp)

                # นับนิ้วที่ยกขึ้น (ยกเว้นนิ้วโป้ง)
                other_fingers_up = ring_up + pinky_up

                # ตรวจสอบว่านิ้วชี้และนิ้วกลางยกขึ้น และนิ้วอื่นไม่ยกขึ้น
                if index_up and middle_up and not thumb_up and other_fingers_up == 0:
                    # คำนวณระยะห่างระหว่างนิ้วชี้และนิ้วกลาง (x-axis)
                    distance = abs(index_tip.x - middle_tip.x)

                    now = time.time()
                    if now - last_scroll_time > scroll_cooldown:
                        if distance < distance_threshold:
                            print("เลื่อนขวา")
                            pyautogui.press('right')  # เลื่อนขวา
                            last_scroll_time = now
                            action = "RIGHT"
                        elif distance > distance_threshold:
                            print("เลื่อนซ้าย")
                            pyautogui.press('left')  # เลื่อนซ้าย
                            last_scroll_time = now
                            action = "LEFT"
                else:
                    action = "NONE"

        # แสดงผล Action บนหน้าจอ
        cv2.putText(frame, action, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # แสดงผลภาพ
        cv2.imshow("Hand Scroll Control", frame)

        # กด ESC เพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == 27:
            break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()