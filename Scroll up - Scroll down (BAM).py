import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

scroll_amount = 40
scroll_cooldown = 0.1

# เปิดกล้อง — ตรวจสอบว่าเปิดได้จริงหรือไม่
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้!")
    print("💡 ลอง:")
    print("   - เปลี่ยนเป็น cv2.VideoCapture(1)")
    print("   - ปิดโปรแกรมอื่นที่ใช้กล้องอยู่ (Zoom, Teams, Camera app)")
    print("   - ตรวจสอบสิทธิ์การเข้าถึงกล้องในระบบ")
    exit()
else:
    print("✅ เปิดกล้องสำเร็จ — กด 'q' เพื่อปิดโปรแกรม")

last_gesture = None
last_scroll_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("❌ ไม่สามารถอ่านภาพจากกล้องได้ — กล้องอาจถูกใช้งานโดยโปรแกรมอื่น")
        time.sleep(1)
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    current_gesture = "stop"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Landmarks ของนิ้วชี้
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            # Landmarks ของนิ้วอื่น ๆ
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

            # ✅ ฟังก์ชันช่วย: คำนวณระยะระหว่างจุด 2 จุด
            def distance(p1, p2):
                return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

            # ✅ ตรวจจับ "กำมือ" แบบแม่นยำ
            index_length = distance(index_mcp, index_tip)
            index_full_length = distance(index_mcp, index_pip) * 2.5

            middle_length = distance(middle_mcp, middle_tip)
            middle_full_length = distance(middle_mcp, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]) * 2.5

            ring_length = distance(ring_mcp, ring_tip)
            ring_full_length = distance(ring_mcp, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]) * 2.5

            pinky_length = distance(pinky_mcp, pinky_tip)
            pinky_full_length = distance(pinky_mcp, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]) * 2.5

            fist_threshold = 0.6

            is_fist = (
                index_length < index_full_length * fist_threshold and
                middle_length < middle_full_length * fist_threshold and
                ring_length < ring_full_length * fist_threshold and
                pinky_length < pinky_full_length * fist_threshold
            )

            if is_fist:
                current_gesture = "stop"
            else:
                # ✅ ตรวจสอบว่านิ้วอื่นพับ
                others_folded = (
                    middle_tip.y > index_pip.y and
                    ring_tip.y > index_pip.y and
                    pinky_tip.y > index_pip.y
                )

                if others_folded:
                    # ✅ ตรวจสอบว่านิ้วชี้เหยียดพอ
                    if index_length > index_full_length * 0.7:
                        dx = index_tip.x - index_pip.x
                        dy = index_tip.y - index_pip.y

                        angle_rad = math.atan2(dx, -dy)
                        angle_deg = math.degrees(angle_rad)
                        if angle_deg < 0:
                            angle_deg += 360

                        if angle_deg < 30 or angle_deg > 330:
                            current_gesture = "scroll_up"
                        elif 150 < angle_deg < 210:
                            current_gesture = "scroll_down"
                        else:
                            current_gesture = "stop"
                    else:
                        current_gesture = "stop"
                else:
                    current_gesture = "stop"

    # 🖱️ Execute scroll
    current_time = time.time()
    if current_gesture != last_gesture or (current_time - last_scroll_time > scroll_cooldown):
        if current_gesture == "scroll_up":
            pyautogui.scroll(scroll_amount)
        elif current_gesture == "scroll_down":
            pyautogui.scroll(-scroll_amount)

        last_gesture = current_gesture
        last_scroll_time = current_time

    # 📊 แสดงเฉพาะสถานะ gesture บนหน้าจอ — ไม่มี debug info
    cv2.putText(image, f"Status: {current_gesture.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 🖼️ แสดงภาพ
    cv2.imshow('Hand Gesture Control', image)

    # บังคับ refresh หน้าต่าง
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
print("👋 ปิดโปรแกรมเรียบร้อย")