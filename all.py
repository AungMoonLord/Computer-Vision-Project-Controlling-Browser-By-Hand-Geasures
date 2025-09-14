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
    min_tracking_confidence=0.7,
    model_complexity=0
)
mp_draw = mp.solutions.drawing_utils

# --- Global variables ---
prev_distance = None
last_zoom_time = 0
last_reset_time = 0
last_scroll_time = 0
last_screenshot_time = 0
last_horizontal_scroll_time = 0
last_gesture = None

# Cooldown settings
zoom_cooldown = 1.0
reset_cooldown = 2.0
scroll_cooldown = 0.1
screenshot_cooldown = 2.0
horizontal_scroll_cooldown = 0.1 #เดิม 0.02

# Other settings
scroll_amount = 40
distance_threshold = 0.05
screenshot_count = 0

# Disable fail-safe
pyautogui.FAILSAFE = False

# --- Helper functions ---
def distance(p1, p2):
    """Calculate distance between two points"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_thumb_and_index_up(landmarks):
    """Check if only thumb and index finger are up (for zoom)"""
    thumb_up = landmarks[4].y < landmarks[3].y
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    return thumb_up and index_up and not (middle_up or ring_up or pinky_up)

def is_all_fingers_up(landmarks):
    """Check if all fingers are up (for zoom reset)"""
    return (landmarks[4].y < landmarks[3].y and
            landmarks[8].y < landmarks[6].y and
            landmarks[12].y < landmarks[10].y and
            landmarks[16].y < landmarks[14].y and
            landmarks[20].y < landmarks[18].y)

def is_three_fingers_up(landmarks):
    """Check if 3 fingers are up (index, middle, ring) for screenshot"""
    try:
        # Check if 3 fingers are up
        index_up = landmarks[8].y < landmarks[6].y   # Index finger
        middle_up = landmarks[12].y < landmarks[10].y  # Middle finger  
        ring_up = landmarks[16].y < landmarks[14].y    # Ring finger
        
        # Check if pinky is down
        pinky_down = landmarks[20].y > landmarks[18].y
        
        # Check if fingers are separated
        index_middle_separated = abs(landmarks[8].x - landmarks[12].x) > 0.03
        middle_ring_separated = abs(landmarks[12].x - landmarks[16].x) > 0.03
        
        return (index_up and middle_up and ring_up and pinky_down and 
                index_middle_separated and middle_ring_separated)
    except:
        return False

def is_two_fingers_horizontal(landmarks):
    """Check if index and middle fingers are up for horizontal scrolling"""
    try:
        # Check finger positions
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        
        # Check if fingers are up
        index_up = index_tip.y < index_mcp.y
        middle_up = middle_tip.y < middle_mcp.y
        
        # Check thumb position (should be down for horizontal scroll)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[1]
        thumb_down = thumb_tip.x > thumb_mcp.x
        
        # Check other fingers are down
        ring_tip = landmarks[16]
        ring_mcp = landmarks[13]
        ring_down = ring_tip.y > ring_mcp.y
        
        pinky_tip = landmarks[20]
        pinky_mcp = landmarks[17]
        pinky_down = pinky_tip.y > pinky_mcp.y
        
        return index_up and middle_up and thumb_down and ring_down and pinky_down
    except:
        return False

def is_hand_closed(landmarks):
    """Check if hand is closed (fist)"""
    try:
        return landmarks[8].y > landmarks[6].y  # Index finger folded
    except:
        return True

def calculate_zoom_gesture(thumb_tip, index_tip):
    """Calculate zoom in/out based on thumb-index distance"""
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
    """Detect vertical scroll gesture"""
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

    # Calculate finger lengths
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

def detect_horizontal_scroll(landmarks):
    """Detect horizontal scroll gesture"""
    global last_horizontal_scroll_time
    
    if not is_two_fingers_horizontal(landmarks):
        return None
    
    current_time = time.time()
    if current_time - last_horizontal_scroll_time < horizontal_scroll_cooldown:
        return None
    
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    
    # Calculate distance between fingers
    finger_distance = abs(index_tip.x - middle_tip.x)
    
    if finger_distance < distance_threshold:
        pyautogui.press('right')
        last_horizontal_scroll_time = current_time
        return "RIGHT"
    elif finger_distance > distance_threshold:
        pyautogui.press('left')
        last_horizontal_scroll_time = current_time
        return "LEFT"
    
    return None

def take_screenshot():
    """Take screenshot when gesture is detected"""
    global last_screenshot_time, screenshot_count
    
    current_time = time.time()
    if current_time - last_screenshot_time > screenshot_cooldown:
        try:
            screenshot = pyautogui.screenshot()
            screenshot_count += 1
            
            filename = f"screenshot_{screenshot_count}_{int(time.time())}.png"
            screenshot.save(filename)
            
            print(f"Screenshot saved: {filename}")
            last_screenshot_time = current_time
            return "SCREENSHOT TAKEN!"
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return None
    return None

# --- Main Program ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้")
    exit()

print("✅ เปิดกล้องสำเร็จ — กด 'q' เพื่อออก")
print("Gestures:")
print("- Thumb + Index: Zoom In/Out")
print("- All fingers up: Reset Zoom")
print("- Index finger pointing: Vertical Scroll")
print("- Index + Middle (horizontal): Left/Right")
print("- 3 fingers up (index+middle+ring): Screenshot")

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

            # Priority 1: Screenshot (3 fingers up)
            if not is_hand_closed(landmarks) and is_three_fingers_up(landmarks):
                screenshot_result = take_screenshot()
                if screenshot_result:
                    status_text = screenshot_result
                continue

            # Priority 2: Horizontal scroll (2 fingers horizontal)
            horizontal_scroll = detect_horizontal_scroll(landmarks)
            if horizontal_scroll:
                status_text = f"Horizontal Scroll {horizontal_scroll}"
                continue

            # Priority 3: Vertical scroll
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

            # Priority 4: Zoom (only if no scroll detected)
            if not scroll_detected:
                if is_thumb_and_index_up(landmarks):
                    zoom_action = calculate_zoom_gesture(landmarks[4], landmarks[8])
                    if zoom_action:
                        status_text = zoom_action

            # Priority 5: Reset Zoom
            if is_all_fingers_up(landmarks) and (current_time - last_reset_time) > reset_cooldown:
                pyautogui.hotkey('ctrl', '0')
                last_reset_time = current_time
                status_text = "Reset Zoom"

    # Display status and instructions
    if status_text:
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display screenshot count
    cv2.putText(frame, f"Screenshots: {screenshot_count}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display instructions
    instructions = [
        "Thumb+Index: Zoom | All fingers: Reset",
        "Point up/down: Scroll | 2 fingers: Left/Right", 
        "3 fingers: Screenshot | Press 'q' to quit"
    ]
    
    y_pos = frame.shape[0] - 80
    for instruction in instructions:
        cv2.putText(frame, instruction, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25

    cv2.imshow("Multi-Gesture Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()