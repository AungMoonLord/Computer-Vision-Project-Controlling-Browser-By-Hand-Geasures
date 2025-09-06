import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np

class GestureZoomController:
    def __init__(self):
        # ตั้งค่า MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # ตัวแปรควบคุม timing และ gesture state
        self.prev_distance = None
        self.last_zoom_time = 0
        self.last_reset_time = 0
        self.zoom_cooldown = 0.1  # ลด cooldown เพื่อให้ responsive มากขึ้น
        self.reset_cooldown = 2.0  # เพิ่ม cooldown สำหรับ reset
        self.gesture_active = False
        self.initial_distance = None
        
        # การปรับปรุงประสิทธิภาพ
        self.distance_buffer = []  # เก็บประวัติระยะห่างเพื่อ smooth การทำงาน
        self.buffer_size = 5
        self.min_distance_change = 0.02  # threshold สำหรับการเปลี่ยนแปลงขั้นต่ำ
        
        # ป้องกัน pyautogui fail-safe
        pyautogui.FAILSAFE = False
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def calculate_distance(self, point1, point2):
        """คำนวณระยะห่างระหว่าง 2 จุด"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def smooth_distance(self, distance):
        """ทำให้ระยะห่างเรียบขึ้นด้วย moving average"""
        self.distance_buffer.append(distance)
        if len(self.distance_buffer) > self.buffer_size:
            self.distance_buffer.pop(0)
        return sum(self.distance_buffer) / len(self.distance_buffer)
    
    def is_thumb_and_index_up(self, landmarks):
        """ตรวจสอบท่าทางนิ้วโป้งและนิ้วชี้ยกขึ้น (ปรับปรุงความแม่นยำ)"""
        try:
            # ใช้ threshold ที่ปรับปรุงแล้ว
            thumb_up = landmarks[4].y < landmarks[3].y - 0.02
            index_up = landmarks[8].y < landmarks[6].y - 0.02
            
            # ตรวจสอบนิ้วอื่น ๆ ให้แม่นยำมากขึ้น
            middle_down = landmarks[12].y > landmarks[10].y + 0.01
            ring_down = landmarks[16].y > landmarks[14].y + 0.01
            pinky_down = landmarks[20].y > landmarks[18].y + 0.01
            
            return thumb_up and index_up and middle_down and ring_down and pinky_down
        except (IndexError, AttributeError):
            return False
    
    def is_all_fingers_up(self, landmarks):
        """ตรวจสอบท่าทางเปิดมือทั้งหมด (ปรับปรุงความแม่นยำ)"""
        try:
            threshold = 0.02
            thumb = landmarks[4].y < landmarks[3].y - threshold
            index = landmarks[8].y < landmarks[6].y - threshold
            middle = landmarks[12].y < landmarks[10].y - threshold
            ring = landmarks[16].y < landmarks[14].y - threshold
            pinky = landmarks[20].y < landmarks[18].y - threshold
            
            return all([thumb, index, middle, ring, pinky])
        except (IndexError, AttributeError):
            return False
    
    def calculate_zoom_gesture(self, thumb_tip, index_tip, landmarks):
        """ควบคุม Zoom In/Out (ปรับปรุงประสิทธิภาพ)"""
        current_time = time.time()
        
        # ตรวจสอบ cooldown
        if (current_time - self.last_zoom_time) < self.zoom_cooldown:
            return None
        
        # ตรวจสอบท่าทางนิ้ว
        if not self.is_thumb_and_index_up(landmarks):
            self.reset_gesture_state()
            return None
        
        # คำนวณระยะห่าง
        distance = self.calculate_distance(thumb_tip, index_tip)
        smoothed_distance = self.smooth_distance(distance)
        
        # เริ่ม gesture ใหม่
        if not self.gesture_active:
            self.gesture_active = True
            self.initial_distance = smoothed_distance
            self.prev_distance = smoothed_distance
            return None
        
        # คำนวณการเปลี่ยนแปลง
        if self.prev_distance is not None:
            diff = smoothed_distance - self.prev_distance
            
            # ใช้ adaptive threshold
            dynamic_threshold = max(0.015, self.initial_distance * 0.08)
            
            if abs(diff) > self.min_distance_change:  # มีการเปลี่ยนแปลงจริง ๆ
                if diff > dynamic_threshold:  # กางออก - Zoom In
                    try:
                        pyautogui.hotkey('ctrl', '+')
                        print(f"Zoom In (diff: {diff:.3f}, distance: {smoothed_distance:.3f})")
                        self.last_zoom_time = current_time
                        self.prev_distance = smoothed_distance
                        return "Zoom In"
                    except Exception as e:
                        print(f"Error in zoom in: {e}")
                        
                elif diff < -dynamic_threshold:  # หุบเข้า - Zoom Out
                    try:
                        pyautogui.hotkey('ctrl', '-')
                        print(f"Zoom Out (diff: {diff:.3f}, distance: {smoothed_distance:.3f})")
                        self.last_zoom_time = current_time
                        self.prev_distance = smoothed_distance
                        return "Zoom Out"
                    except Exception as e:
                        print(f"Error in zoom out: {e}")
        
        return None
    
    def reset_gesture_state(self):
        """รีเซ็ตสถานะ gesture"""
        self.gesture_active = False
        self.prev_distance = None
        self.initial_distance = None
        self.distance_buffer.clear()
    
    def handle_reset_zoom(self, landmarks):
        """จัดการการรีเซ็ตซูม"""
        current_time = time.time()
        if (self.is_all_fingers_up(landmarks) and 
            (current_time - self.last_reset_time) > self.reset_cooldown):
            try:
                pyautogui.hotkey('ctrl', '0')
                print("Reset Zoom to 100%")
                self.last_reset_time = current_time
                self.reset_gesture_state()  # รีเซ็ต gesture state ด้วย
                return True
            except Exception as e:
                print(f"Error in reset zoom: {e}")
        return False
    
    def update_fps(self):
        """อัพเดท FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            print(f"FPS: {self.fps_counter}")
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def draw_info(self, image, landmarks):
        """วาดข้อมูลบนหน้าจอ"""
        height, width = image.shape[:2]
        
        # วาด FPS
        cv2.putText(image, f"FPS: {self.fps_counter}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # วาดสถานะ gesture
        if self.gesture_active and self.is_thumb_and_index_up(landmarks):
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            distance = self.calculate_distance(thumb_tip, index_tip)
            
            cv2.putText(image, f"Distance: {distance:.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(image, "Zoom Gesture Active", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # วาด instructions
        cv2.putText(image, "Thumb+Index: Zoom | Open Hand: Reset", (10, height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """รันโปรแกรมหลัก"""
        cap = cv2.VideoCapture(0)
        
        # ตั้งค่ากล้องสำหรับประสิทธิภาพที่ดีขึ้น
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # จำกัดให้ตรวจจับมือเดียว
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:
            
            print("Hand Gesture Zoom Control Started!")
            print("Controls:")
            print("- Thumb + Index finger: Zoom in/out")
            print("- Open hand (5 fingers): Reset zoom")
            print("- Press 'q' to quit")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read frame")
                    break
                
                # พลิกภาพให้เป็น mirror
                frame = cv2.flip(frame, 1)
                
                # แปลงสีสำหรับ MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # วาดผลลัพธ์
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # วาด landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        landmarks = hand_landmarks.landmark
                        
                        # จัดการ zoom gesture
                        if self.is_thumb_and_index_up(landmarks):
                            thumb_tip = landmarks[4]
                            index_tip = landmarks[8]
                            zoom_action = self.calculate_zoom_gesture(thumb_tip, index_tip, landmarks)
                            
                            if zoom_action:
                                cv2.putText(frame, zoom_action, (10, 120),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        else:
                            # รีเซ็ต gesture เมื่อไม่ได้ทำท่า
                            if self.gesture_active:
                                self.reset_gesture_state()
                        
                        # จัดการ reset zoom
                        if self.handle_reset_zoom(landmarks):
                            cv2.putText(frame, "Reset Zoom!", (10, 150),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # วาดข้อมูล
                        self.draw_info(frame, landmarks)
                else:
                    # รีเซ็ต state เมื่อไม่เจอมือ
                    if self.gesture_active:
                        self.reset_gesture_state()
                
                # แสดงผล
                cv2.imshow('Hand Gesture Zoom Control', frame)
                
                # อัพเดท FPS
                self.update_fps()
                
                # ตรวจสอบการกดปุ่ม
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' หรือ ESC
                    break
        
        # ปิดทรัพยากร
        cap.release()
        cv2.destroyAllWindows()
        print("Program terminated successfully")

# รันโปรแกรม
if __name__ == "__main__":
    controller = GestureZoomController()
    controller.run()
