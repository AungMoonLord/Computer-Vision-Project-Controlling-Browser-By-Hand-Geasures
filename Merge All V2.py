import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

class HandGestureController:
    def __init__(self):
        # ตั้งค่า MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0
        )
        
        # เปิดกล้อง
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # ตัวแปรควบคุม timing
        self.last_action_time = 0
        self.last_screenshot_time = 0
        self.last_zoom_time = 0
        self.last_reset_time = 0
        
        # Cooldown periods
        self.scroll_cooldown = 0.1
        self.screenshot_cooldown = 2.0
        self.zoom_cooldown = 0.5
        self.reset_cooldown = 2.0
        
        # ตัวแปรสำหรับ gesture tracking
        self.prev_zoom_distance = None
        self.screenshot_count = 0
        self.current_gesture = "NONE"
        
        # ตัวแปรสำหรับ scroll ซ้าย-ขวา
        self.distance_threshold = 0.05
        
        # ตัวแปรสำหรับ zoom
        self.zoom_threshold = 0.15
        
        # ปิด fail-safe
        pyautogui.FAILSAFE = False
        
        print("✅ Hand Gesture Controller เริ่มทำงาน")
        print("📝 คำแนะนำการใช้งาน:")
        print("   1 นิ้ว (นิ้วชี้) → Scroll Up/Down")
        print("   2 นิ้ว (ชี้+กลาง) → Scroll Left/Right")
        print("   3 นิ้ว (ชี้+กลาง+นาง) → Screenshot")
        print("   นิ้วโป้ง+ชี้ → Zoom In/Out")
        print("   5 นิ้ว → Reset Zoom")
        print("   กด 'q' เพื่อออกจากโปรแกรม")

    def calculate_distance(self, point1, point2):
        """คำนวณระยะห่างระหว่าง 2 จุด"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def is_finger_up(self, tip, mcp):
        """ตรวจสอบว่านิ้วยกขึ้นหรือไม่"""
        return tip.y < mcp.y

    def detect_one_finger(self, landmarks):
        """ตรวจจับนิ้วชี้เดียว สำหรับ scroll up/down"""
        try:
            # ตรวจสอบนิ้วแต่ละนิ้ว
            index_up = self.is_finger_up(landmarks[8], landmarks[5])
            middle_up = self.is_finger_up(landmarks[12], landmarks[9])
            ring_up = self.is_finger_up(landmarks[16], landmarks[13])
            pinky_up = self.is_finger_up(landmarks[20], landmarks[17])
            thumb_up = landmarks[4].x < landmarks[3].x
            
            # เฉพาะนิ้วชี้ยกขึ้น
            if index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
                # ตรวจสอบทิศทางของนิ้วชี้
                index_tip = landmarks[8]
                index_pip = landmarks[6]
                
                # คำนวณมุม
                dx = index_tip.x - index_pip.x
                dy = index_tip.y - index_pip.y
                angle_rad = math.atan2(dx, -dy)
                angle_deg = math.degrees(angle_rad)
                if angle_deg < 0:
                    angle_deg += 360
                
                # ชี้ขึ้น
                if angle_deg < 30 or angle_deg > 330:
                    return "scroll_up"
                # ชี้ลง
                elif 150 < angle_deg < 210:
                    return "scroll_down"
            
            return None
        except:
            return None

    def detect_two_fingers(self, landmarks):
        """ตรวจจับ 2 นิ้ว (ชี้+กลาง) สำหรับ scroll left/right"""
        try:
            index_up = self.is_finger_up(landmarks[8], landmarks[5])
            middle_up = self.is_finger_up(landmarks[12], landmarks[9])
            ring_up = self.is_finger_up(landmarks[16], landmarks[13])
            pinky_up = self.is_finger_up(landmarks[20], landmarks[17])
            thumb_up = landmarks[4].x < landmarks[3].x
            
            # เฉพาะนิ้วชี้และนิ้วกลางยกขึ้น
            if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                
                # คำนวณระยะห่างระหว่างนิ้วชี้และนิ้วกลาง
                distance = abs(index_tip.x - middle_tip.x)
                
                if distance < self.distance_threshold:
                    return "scroll_right"
                elif distance > self.distance_threshold * 2:
                    return "scroll_left"
            
            return None
        except:
            return None

    def detect_three_fingers(self, landmarks):
        """ตรวจจับ 3 นิ้ว (ชี้+กลาง+นาง) สำหรับ screenshot"""
        try:
            index_up = self.is_finger_up(landmarks[8], landmarks[6])
            middle_up = self.is_finger_up(landmarks[12], landmarks[10])
            ring_up = self.is_finger_up(landmarks[16], landmarks[14])
            pinky_up = self.is_finger_up(landmarks[20], landmarks[18])
            
            # ตรวจสอบว่านิ้วที่ชี้ขึ้นมีการแยกออกจากกัน
            index_middle_separated = abs(landmarks[8].x - landmarks[12].x) > 0.03
            middle_ring_separated = abs(landmarks[12].x - landmarks[16].x) > 0.03
            
            return (index_up and middle_up and ring_up and not pinky_up and 
                   index_middle_separated and middle_ring_separated)
        except:
            return False

    def detect_thumb_index(self, landmarks):
        """ตรวจจับนิ้วโป้ง+นิ้วชี้ สำหรับ zoom"""
        try:
            thumb_up = landmarks[4].y < landmarks[3].y
            index_up = self.is_finger_up(landmarks[8], landmarks[6])
            middle_up = self.is_finger_up(landmarks[12], landmarks[10])
            ring_up = self.is_finger_up(landmarks[16], landmarks[14])
            pinky_up = self.is_finger_up(landmarks[20], landmarks[18])
            
            # เฉพาะนิ้วโป้งและนิ้วชี้
            other_fingers_up = middle_up or ring_up or pinky_up
            return thumb_up and index_up and not other_fingers_up
        except:
            return False

    def detect_five_fingers(self, landmarks):
        """ตรวจจับ 5 นิ้ว สำหรับ reset zoom"""
        try:
            thumb = landmarks[4].y < landmarks[3].y
            index = self.is_finger_up(landmarks[8], landmarks[6])
            middle = self.is_finger_up(landmarks[12], landmarks[10])
            ring = self.is_finger_up(landmarks[16], landmarks[14])
            pinky = self.is_finger_up(landmarks[20], landmarks[18])
            
            return thumb and index and middle and ring and pinky
        except:
            return False

    def handle_zoom_gesture(self, landmarks):
        """จัดการ zoom in/out"""
        current_time = time.time()
        if (current_time - self.last_zoom_time) < self.zoom_cooldown:
            return None
            
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = self.calculate_distance(thumb_tip, index_tip)
        
        if self.prev_zoom_distance is not None:
            # เพิ่มระยะห่าง → Zoom In
            if distance > self.prev_zoom_distance + 0.02:
                pyautogui.hotkey('ctrl', '+')
                self.last_zoom_time = current_time
                self.prev_zoom_distance = distance
                return "ZOOM IN"
            # ลดระยะห่าง → Zoom Out
            elif distance < self.prev_zoom_distance - 0.02:
                pyautogui.hotkey('ctrl', '-')
                self.last_zoom_time = current_time
                self.prev_zoom_distance = distance
                return "ZOOM OUT"
        else:
            self.prev_zoom_distance = distance
            
        return None

    def take_screenshot(self):
        """ถ่าย screenshot"""
        try:
            screenshot = pyautogui.screenshot()
            self.screenshot_count += 1
            filename = f"screenshot_{self.screenshot_count}_{int(time.time())}.png"
            screenshot.save(filename)
            print(f"📸 Screenshot saved: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error taking screenshot: {e}")
            return False

    def process_gestures(self, landmarks):
        """ประมวลผล gestures ทั้งหมด - แก้ไข Priority และการตรวจจับ"""
        current_time = time.time()
        action_taken = None
        
        # Priority 1: Reset Zoom (5 นิ้ว) - ตรวจก่อนเพราะครอบคลุมทุกนิ้ว
        if self.detect_five_fingers(landmarks):
            if (current_time - self.last_reset_time) > self.reset_cooldown:
                pyautogui.hotkey('ctrl', '0')
                self.last_reset_time = current_time
                action_taken = "RESET ZOOM"
        
        # Priority 2: Screenshot (3 นิ้ว) - ตรวจก่อน thumb+index
        elif self.detect_three_fingers(landmarks):
            if (current_time - self.last_screenshot_time) > self.screenshot_cooldown:
                if self.take_screenshot():
                    self.last_screenshot_time = current_time
                    action_taken = "SCREENSHOT"
        
        # Priority 3: Zoom (นิ้วโป้ง + นิ้วชี้) - ตรวจก่อน 2 finger และ 1 finger
        elif self.detect_thumb_index(landmarks):
            zoom_action = self.handle_zoom_gesture(landmarks)
            if zoom_action:
                action_taken = zoom_action
        
        # Priority 4: Scroll Left/Right (2 นิ้ว) - ตรวจก่อน 1 finger
        elif self.detect_two_fingers(landmarks):
            if (current_time - self.last_action_time) > self.scroll_cooldown:
                scroll_action = self.detect_two_fingers(landmarks)
                if scroll_action == "scroll_left":
                    pyautogui.press('left')
                    action_taken = "SCROLL LEFT"
                elif scroll_action == "scroll_right":
                    pyautogui.press('right')
                    action_taken = "SCROLL RIGHT"
                
                if action_taken:
                    self.last_action_time = current_time
        
        # Priority 5: Scroll Up/Down (1 นิ้ว) - ตรวจท้ายสุด
        else:
            scroll_action = self.detect_one_finger(landmarks)
            if scroll_action and (current_time - self.last_action_time) > self.scroll_cooldown:
                if scroll_action == "scroll_up":
                    pyautogui.scroll(3)
                    action_taken = "SCROLL UP"
                elif scroll_action == "scroll_down":
                    pyautogui.scroll(-3)
                    action_taken = "SCROLL DOWN"
                
                if action_taken:
                    self.last_action_time = current_time
        
        return action_taken if action_taken else "READY"

    def draw_ui(self, frame):
        """วาด UI แสดงข้อมูลบนหน้าจอ"""
        height, width = frame.shape[:2]
        
        # พื้นหลังสำหรับข้อมูล
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # แสดงสถานะปัจจุบัน
        cv2.putText(frame, f"Status: {self.current_gesture}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # แสดงจำนวน screenshots
        cv2.putText(frame, f"Screenshots: {self.screenshot_count}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # แสดงคำแนะนำ
        cv2.putText(frame, "Press 'q' to quit", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # แสดงคำแนะนำ gestures
        instructions = [
            "1 finger: Scroll Up/Down",
            "2 fingers: Scroll Left/Right", 
            "3 fingers: Screenshot",
            "Thumb+Index: Zoom In/Out",
            "5 fingers: Reset Zoom"
        ]
        
        start_y = height - 120
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, start_y + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def run(self):
        """เริ่มการทำงานหลัก"""
        if not self.cap.isOpened():
            print("❌ ไม่สามารถเปิดกล้องได้!")
            return
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("❌ ไม่สามารถอ่านภาพจากกล้องได้")
                break
            
            # กลับด้านภาพ
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ประมวลผล hand detection
            results = self.hands.process(rgb_frame)
            
            self.current_gesture = "NO HAND"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # วาด landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # ประมวลผล gestures
                    landmarks = hand_landmarks.landmark
                    self.current_gesture = self.process_gestures(landmarks)
            
            # วาด UI
            self.draw_ui(frame)
            
            # แสดงผล
            cv2.imshow('Hand Gesture Controller - All Features', frame)
            
            # ตรวจสอบการกดปุ่ม
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cleanup()

    def cleanup(self):
        """ปิดการทำงานและล้างทรัพยากร"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("👋 ปิดโปรแกรมเรียบร้อย")

if __name__ == "__main__":
    controller = HandGestureController()
    controller.run()
    