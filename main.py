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

# Scroll settings
scroll_amount = 40
scroll_cooldown = 0.1

# Zoom settings
prev_distance = None
last_zoom_time = 0
last_reset_time = 0
zoom_cooldown = 1
reset_cooldown = 2

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้!")
    exit()
else:
    print("✅ เปิดกล้องสำเร็จ — กด 'q' เพื่อปิดโปรแกรม")

last_gesture = None
last_scroll_time = 0

# Helper functions
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_fist(landmarks):
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]

    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    index_length = distance(index_mcp, index_tip)
    index_full_length = distance(index_mcp, index_pip) * 2.5

    middle_length = distance(middle_mcp, middle_tip)
    ring_length = distance(ring_mcp, ring_tip)
    pinky_length = distance(pinky_mcp, pinky_tip)

    fist_threshold = 0.6
    return (
        index_length < index_full_length * fist_threshold and
        middle_length < index_full_length * fist_threshold and
        ring_length < index_full_length * fist_threshold and
        pinky_length < index_full_length * fist_threshold
    )

def is_scroll_gesture(landmarks):
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    index_length = distance(index_mcp, index_tip)
    index_full_length = distance(index_mcp, index_pip) * 2.5

    others_folded = (
        middle_tip.y > index_pip.y and
        ring_tip.y > index_pip.y and
        pinky_tip.y > index_pip.y
    )

    if others_folded and index_length > index_full_length * 0.7:
        dx = index_tip.x - index_pip.x
        dy = index_tip.y - index_pip.y
        angle_rad = math.atan2(dx, -dy)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        if angle_deg < 30 or angle_deg > 330:
            return "scroll_up"
        elif 150 < angle_deg < 210:
            return "scroll_down"

    return "stop"

def calculate_zoom_gesture(thumb_tip, index_tip):
    global prev_distance, last_zoom_time

    current_time = time.time()
    dist = distance(thumb_tip, index_tip)

    if (current_time - last_zoom_time) < zoom_cooldown:
        return None

    if prev_distance is not None:
        diff = dist - prev_distance
        if diff > 0.03:
            pyautogui.hotkey('ctrl', '+')
            last_zoom_time = current_time
            return "Zoom In"
        elif diff < -0.03:
            pyautogui.hotkey('ctrl', '-')
            last_zoom_time = current_time
            return "Zoom Out"

    prev_distance = dist
    return None

def is_all_fingers_up(landmarks):
    thumb = landmarks[4].y < landmarks[3].y
    index = landmarks[8].y < landmarks[6].y
    middle = landmarks[12].y < landmarks[10].y
    ring = landmarks[16].y < landmarks[14].y
    pinky = landmarks[20].y < landmarks[18].y
    return thumb and index and middle and ring and pinky

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("❌ ไม่สามารถอ่านภาพจากกล้องได้")
        time.sleep(1)
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    current_gesture = "stop"
    zoom_action = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Scroll gesture
            if is_fist(hand_landmarks):
                current_gesture = "stop"
            else:
                current_gesture = is_scroll_gesture(landmarks)

            # Zoom gesture
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            zoom_action = calculate_zoom_gesture(thumb_tip, index_tip)

            # Reset zoom gesture
            current_time = time.time()
            if is_all_fingers_up(landmarks) and (current_time - last_reset_time) > reset_cooldown:
                pyautogui.hotkey('ctrl', '0')
                last_reset_time = current_time
                cv2.putText(image, "Reset Zoom", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show distance
            dist = distance(thumb_tip, index_tip)
            cv2.putText(image, f"Distance: {dist:.2f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show zoom action
            if zoom_action:
                cv2.putText(image, zoom_action, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Execute scroll
    current_time = time.time()
    if current_gesture != last_gesture or (current_time - last_scroll_time > scroll_cooldown):
        if current_gesture == "scroll_up":
            pyautogui.scroll(scroll_amount)
        elif current_gesture == "scroll_down":
            pyautogui.scroll(-scroll_amount)

        last_gesture = current_gesture
        last_scroll_time = current_time

    # Display status
    cv2.putText(image, f"Status: {current_gesture.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show image
    cv2.imshow('Hand Gesture Control', image)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("👋 ปิดโปรแกรมเรียบร้อย")

"""รวมโค้ด AUNG & BAM เรียบร้อยแล้ว"""