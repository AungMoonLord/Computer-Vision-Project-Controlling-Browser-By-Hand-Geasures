import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- Global variables ---
prev_distance = None
last_zoom_time = 0
last_reset_time = 0
last_scroll_time = 0
last_gesture = None

zoom_cooldown = 1.0
reset_cooldown = 2.0
scroll_cooldown = 0.1
scroll_amount = 40

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

    threshold = 0.18
    if prev_distance is not None:
        if dist > threshold and dist < 0.27:
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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            current_time = time.time()

            # Zoom
            if is_thumb_and_index_up(landmarks):
                zoom_action = calculate_zoom_gesture(landmarks[4], landmarks[8])
                if zoom_action:
                    status_text = zoom_action

            # Reset Zoom
            if is_all_fingers_up(landmarks) and (current_time - last_reset_time) > reset_cooldown:
                pyautogui.hotkey('ctrl', '0')
                last_reset_time = current_time
                status_text = "Reset Zoom"

            # Scroll
            scroll_gesture = detect_scroll_gesture(landmarks)
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

    # Show status
    if status_text:
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
