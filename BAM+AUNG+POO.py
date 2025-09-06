import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ============ ตั้งค่า Mediapipe ============
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============ ตัวแปรควบคุม ============
prev_distance = None
last_zoom_time = 0
last_reset_time = 0
last_scroll_time = 0
last_screenshot_time = 0
last_left_right_time = 0

screenshot_count = 0

# cooldown
zoom_cooldown = 1.0
reset_cooldown = 2.0
scroll_cooldown = 0.2
screenshot_cooldown = 2.0
leftright_cooldown = 0.3

scroll_amount = 40
distance_threshold = 0.05  # สำหรับ left/right

# ปิด fail-safe
pyautogui.FAILSAFE = False

# ============ Helper Functions ============
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def is_thumb_and_index_up(landmarks):
    return landmarks[4].y < landmarks[3].y and landmarks[8].y < landmarks[6].y

def is_all_fingers_up(landmarks):
    return (landmarks[4].y < landmarks[3].y and
            landmarks[8].y < landmarks[6].y and
            landmarks[12].y < landmarks[10].y and
            landmarks[16].y < landmarks[14].y and
            landmarks[20].y < landmarks[18].y)

def is_three_fingers_up(landmarks):
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y
    return index_up and middle_up and ring_up and pinky_down

def is_hand_closed(landmarks):
    return landmarks[8].y > landmarks[6].y

# ============ กล้อง ============
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้")
    exit()

print("✅ เปิดกล้องสำเร็จ — กด 'q' เพื่อออก")

# ============ Main Loop ============
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
            lm = hand_landmarks.landmark

            # ---- Zoom In/Out ----
            if is_thumb_and_index_up(lm) and (current_time - last_zoom_time) > zoom_cooldown:
                dist = distance(lm[4], lm[8])
                if dist > 0.20:
                    pyautogui.hotkey('ctrl', '+')
                    status_text = "Zoom In"
                else:
                    pyautogui.hotkey('ctrl', '-')
                    status_text = "Zoom Out"
                last_zoom_time = current_time

            # ---- Reset Zoom ----
            if is_all_fingers_up(lm) and (current_time - last_reset_time) > reset_cooldown:
                pyautogui.hotkey('ctrl', '0')
                status_text = "Reset Zoom"
                last_reset_time = current_time

            # ---- Scroll Up/Down ----
            index_tip, index_pip = lm[8], lm[6]
            if index_tip.y < index_pip.y and (current_time - last_scroll_time) > scroll_cooldown:
                dx = index_tip.x - index_pip.x
                dy = index_tip.y - index_pip.y
                angle = math.degrees(math.atan2(dx, -dy))
                if angle < 0: angle += 360
                if angle < 30 or angle > 330:
                    pyautogui.scroll(scroll_amount)
                    status_text = "Scroll Up"
                    last_scroll_time = current_time
                elif 150 < angle < 210:
                    pyautogui.scroll(-scroll_amount)
                    status_text = "Scroll Down"
                    last_scroll_time = current_time

            # ---- Screenshot ----
            if not is_hand_closed(lm) and is_three_fingers_up(lm) and (current_time - last_screenshot_time) > screenshot_cooldown:
                screenshot = pyautogui.screenshot()
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}_{int(time.time())}.png"
                screenshot.save(filename)
                status_text = f"Screenshot Saved ({screenshot_count})"
                last_screenshot_time = current_time

            # ---- Left / Right ----
            index_tip, middle_tip = lm[8], lm[12]
            index_mcp, middle_mcp = lm[5], lm[9]
            if (index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y) and (current_time - last_left_right_time) > leftright_cooldown:
                distance_x = abs(index_tip.x - middle_tip.x)
                if distance_x < distance_threshold:
                    pyautogui.press('right')
                    status_text = "Right"
                else:
                    pyautogui.press('left')
                    status_text = "Left"
                last_left_right_time = current_time

    # แสดงผลบนจอ
    if status_text:
        cv2.putText(frame, status_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Hand Gesture Control (Merged)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
