"""ZOOM -> ใช้นิ้วชี้,นิ้วกลาง,นิ้วโป้ง"""
"""Scroll UP/DOWN -> ใช้นิ้วชี้"""

import cv2
import mediapipe as mp
import pyautogui
import time
import math

# -----------------------------
# ตั้งค่า MediaPipe
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# ตัวแปรควบคุมเวลา / สถานะ
# -----------------------------
prev_distance = None
last_zoom_time = 0
last_reset_time = 0
last_scroll_time = 0
last_scroll_gesture = None

zoom_cooldown = 1.0
reset_cooldown = 2.0
scroll_cooldown = 0.1
scroll_amount = 40

# -----------------------------
# ฟังก์ชันช่วย
# -----------------------------
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_all_fingers_up(landmarks):
    return (
        landmarks[4].y < landmarks[3].y and
        landmarks[8].y < landmarks[6].y and
        landmarks[12].y < landmarks[10].y and
        landmarks[16].y < landmarks[14].y and
        landmarks[20].y < landmarks[18].y
    )

def is_thumb_index_middle_up(landmarks):
    """ตรวจว่านิ้วโป้ง + นิ้วชี้ + นิ้วกลางยกขึ้น และนิ้วนาง/ก้อยไม่ยก"""
    thumb_up = landmarks[4].y < landmarks[3].y
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    return thumb_up and index_up and middle_up and not (ring_up or pinky_up)

# -----------------------------
# ตรวจจับ Zoom gesture
# -----------------------------
def detect_zoom_gesture(landmarks):
    global prev_distance, last_zoom_time
    current_time = time.time()

    thumb_tip = landmarks[4]
    middle_tip = landmarks[12]
    dist = distance(thumb_tip, middle_tip)

    if current_time - last_zoom_time < zoom_cooldown:
        return None

    if prev_distance is not None:
        if dist > prev_distance + 0.05:  # ยืดออก = Zoom In
            pyautogui.hotkey('ctrl', '+')
            last_zoom_time = current_time
            prev_distance = dist
            return "Zoom In"
        elif dist < prev_distance - 0.05:  # หุบเข้า = Zoom Out
            pyautogui.hotkey('ctrl', '-')
            last_zoom_time = current_time
            prev_distance = dist
            return "Zoom Out"

    prev_distance = dist
    return None

# -----------------------------
# ตรวจจับ Scroll gesture
# -----------------------------
def detect_scroll_gesture(landmarks):
    global last_scroll_time, last_scroll_gesture

    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    def finger_len(mcp, tip, pip):
        return distance(mcp, tip), distance(mcp, pip) * 2.5

    index_len, index_full = finger_len(index_mcp, index_tip, index_pip)
    middle_len, middle_full = finger_len(middle_mcp, middle_tip, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP])
    ring_len, ring_full = finger_len(ring_mcp, ring_tip, landmarks[mp_hands.HandLandmark.RING_FINGER_PIP])
    pinky_len, pinky_full = finger_len(pinky_mcp, pinky_tip, landmarks[mp_hands.HandLandmark.PINKY_PIP])

    fist_threshold = 0.6
    is_fist = (
        index_len < index_full * fist_threshold and
        middle_len < middle_full * fist_threshold and
        ring_len < ring_full * fist_threshold and
        pinky_len < pinky_full * fist_threshold
    )

    if is_fist:
        return "stop"

    others_folded = (
        middle_tip.y > index_pip.y and
        ring_tip.y > index_pip.y and
        pinky_tip.y > index_pip.y
    )

    if others_folded and index_len > index_full * 0.7:
        dx = index_tip.x - index_pip.x
        dy = index_tip.y - index_pip.y
        angle_rad = math.atan2(dx, -dy)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        current_time = time.time()
        gesture = None
        if angle_deg < 30 or angle_deg > 330:
            gesture = "scroll_up"
        elif 150 < angle_deg < 210:
            gesture = "scroll_down"
        else:
            gesture = "stop"

        if gesture != last_scroll_gesture or (current_time - last_scroll_time > scroll_cooldown):
            if gesture == "scroll_up":
                pyautogui.scroll(scroll_amount)
            elif gesture == "scroll_down":
                pyautogui.scroll(-scroll_amount)
            last_scroll_gesture = gesture
            last_scroll_time = current_time

        return gesture

    return "stop"

# -----------------------------
# Main Loop
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้")
    exit()

print("✅ เริ่มทำงาน — กด 'q' เพื่อออก")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    action_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # ✅ Zoom เฉพาะเมื่อชูโป้ง+ชี้+กลาง
            if is_thumb_index_middle_up(landmarks):
                zoom_action = detect_zoom_gesture(landmarks)
                if zoom_action:
                    action_text = zoom_action

            # ✅ Reset Zoom (ชู 5 นิ้ว)
            current_time = time.time()
            if is_all_fingers_up(landmarks) and (current_time - last_reset_time) > reset_cooldown:
                pyautogui.hotkey('ctrl', '0')
                last_reset_time = current_time
                action_text = "Reset Zoom"

            # ✅ Scroll
            scroll_action = detect_scroll_gesture(landmarks)
            if scroll_action and scroll_action != "stop":
                action_text = scroll_action

    if action_text:
        cv2.putText(frame, action_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Hand Gesture Control (Zoom + Scroll)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 ปิดโปรแกรมเรียบร้อย")
