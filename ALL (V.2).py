import cv2
import mediapipe as mp
import pyautogui
import math
import time


# ปิด fail-safe ของ pyautogui
pyautogui.FAILSAFE = False

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)
mp_draw = mp.solutions.drawing_utils

# --- Global variables ---
prev_distance = None
last_zoom_time = 0
last_reset_time = 0
last_scroll_time = 0
last_gesture = None
last_screenshot_time = 0
screenshot_count = 0

zoom_cooldown = 1.0
reset_cooldown = 2.0
scroll_cooldown = 0.1
scroll_amount = 40
screenshot_delay = 2.0

# --- Helper functions ---
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_thumb_and_index_up(landmarks):
    thumb_up = landmarks[4].y < landmarks[3].y
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    return thumb_up and index_up and not (middle_up or ring_up or pinky_up)

def is_all_fingers_up(landmarks):
    return (landmarks[4].y < landmarks[3].y and
            landmarks[8].y < landmarks[6].y and
            landmarks[12].y < landmarks[10].y and
            landmarks[16].y < landmarks[14].y and
            landmarks[20].y < landmarks[18].y)

def calculate_zoom_gesture(thumb_tip, index_tip):
    global prev_distance, last_zoom_time
    current_time = time.time()
    dist = distance(thumb_tip, index_tip)

    if (current_time - last_zoom_time) < zoom_cooldown:
        return None

    threshold = 0.13
    if prev_distance is not None:
        if dist > threshold:
            pyautogui.hotkey('ctrl', '+')
            last_zoom_time = current_time
            return "Zoom In"
        elif dist < threshold:
            pyautogui.hotkey('ctrl', '-')
            last_zoom_time = current_time
            return "Zoom Out"
    prev_distance = dist
    return None

def detect_scroll_gesture(landmarks):
    global last_gesture, last_scroll_time
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    # Lengths
    def length(mcp, tip, pip):
        return distance(mcp, tip), distance(mcp, pip) * 2.5

    index_len, index_full = length(index_mcp, index_tip, index_pip)
    middle_len, middle_full = length(middle_mcp, middle_tip, landmarks[10])
    ring_len, ring_full = length(ring_mcp, ring_tip, landmarks[14])
    pinky_len, pinky_full = length(pinky_mcp, pinky_tip, landmarks[18])

    fist_threshold = 0.6
    is_fist = (index_len < index_full * fist_threshold and
               middle_len < middle_full * fist_threshold and
               ring_len < ring_full * fist_threshold and
               pinky_len < pinky_full * fist_threshold)

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

        if angle_deg < 30 or angle_deg > 330:
            return "scroll_up"
        elif 150 < angle_deg < 210:
            return "scroll_down"
    return "stop"

def is_three_fingers_up(hand_landmarks):
    landmarks = hand_landmarks.landmark
    try:
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_down = landmarks[20].y > landmarks[18].y
        index_middle_separated = abs(landmarks[8].x - landmarks[12].x) > 0.03
        middle_ring_separated = abs(landmarks[12].x - landmarks[16].x) > 0.03
        return (index_up and middle_up and ring_up and pinky_down and
                index_middle_separated and middle_ring_separated)
    except:
        return False

def is_hand_closed(hand_landmarks):
    try:
        landmarks = hand_landmarks.landmark
        index_folded = landmarks[8].y > landmarks[6].y
        return index_folded
    except:
        return True

# --- Main Program ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้")
    exit()

print("✅ เปิดกล้องสำเร็จ — กด 'q' เพื่อออก")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    status_text = ""
    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Scroll Gesture Detection
            scroll_gesture = detect_scroll_gesture(landmarks)
            scroll_detected = False

            if scroll_gesture in ["scroll_up", "scroll_down"]:
                if (scroll_gesture != last_gesture) or (current_time - last_scroll_time > scroll_cooldown):
                    if scroll_gesture == "scroll_up":
                        pyautogui.scroll(scroll_amount)
                        status_text = "Scroll Up"
                    elif scroll_gesture == "scroll_down":
                        pyautogui.scroll(-scroll_amount)
                        status_text = "Scroll Down"
                    last_gesture = scroll_gesture
                    last_scroll_time = current_time
                    scroll_detected = True

            # Zoom Gesture Detection
            if not scroll_detected:
                if is_thumb_and_index_up(landmarks):
                    zoom_action = calculate_zoom_gesture(landmarks[4], landmarks[8])
                    if zoom_action:
                        status_text = zoom_action

            # Reset Zoom
            if is_all_fingers_up(landmarks) and (current_time - last_reset_time) > reset_cooldown:
                pyautogui.hotkey('ctrl', '0')
                last_reset_time = current_time
                status_text = "Reset Zoom"

            # Screenshot Capture
            if not is_hand_closed(hand_landmarks) and is_three_fingers_up(hand_landmarks):
                if current_time - last_screenshot_time > screenshot_delay:
                    try:
                        screenshot = pyautogui.screenshot()
                        screenshot_count += 1
                        filename = f"screenshot_{screenshot_count}_{int(time.time())}.png"
                        screenshot.save(filename)
                        print(f"Screenshot saved: {filename}")
                        last_screenshot_time = current_time
                        status_text = "SCREENSHOT TAKEN"
                    except Exception as e:
                        print(f"Error taking screenshot: {e}")

            # Slide Control (Left/Right)
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            index_mcp = landmarks[5]
            middle_mcp = landmarks[9]
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[1]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            def is_finger_up(tip, mcp):
                return tip.y < mcp.y

            index_up = is_finger_up(index_tip, index_mcp)
            middle_up = is_finger_up(middle_tip, middle_mcp)
            thumb_up = thumb_tip.x < thumb_mcp.x
            ring_up = is_finger_up(ring_tip, landmarks[13])
            pinky_up = is_finger_up(pinky_tip, landmarks[17])

            other_fingers_up = ring_up + pinky_up

            if index_up and middle_up and not thumb_up and other_fingers_up == 0:
                distance_x = abs(index_tip.x - middle_tip.x)
                if current_time - last_scroll_time > scroll_cooldown:
                    if distance_x < 0.05:
                        pyautogui.press('right')
                        status_text = "RIGHT"
                    elif distance_x > 0.05:
                        pyautogui.press('left')
                        status_text = "LEFT"
                    last_scroll_time = current_time

    # Show status
    if status_text:
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display Instructions
    cv2.putText(frame, "Show 3 fingers to take screenshot", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Screenshots: {screenshot_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()