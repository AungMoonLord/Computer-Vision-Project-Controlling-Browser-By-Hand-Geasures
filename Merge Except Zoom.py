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
            min_tracking_confidence=0.6,
            model_complexity=0
        )
        
        # เปิดกล้อง
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ ไม่สามารถเปิดกล้องได้!")
            exit()
        
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # ตัวแปรควบคุมการทำงาน
        self.reset_timers()
        self.screenshot_count = 0
        
        # การตั้งค่า
        self.screenshot_delay = 2.0
        self.scroll_cooldown = 0.1
        self.horizontal_scroll_cooldown = 0.02
        self.scroll_amount = 40
        self.distance_threshold = 0.05
        
        # ปิด fail-safe
        pyautogui.FAILSAFE = False
        
        print("✅ เปิดระบบควบคุมด้วยท่าทางมือสำเร็จ")
        print("📖 คำแนะนำ:")
        print("   🤚 3 นิ้วชี้ขึ้น (ชี้,กลาง,นาง) = ถ่าย Screenshot")
        print("   👆 นิ้วชี้ขึ้น = Scroll Up/Down")
        print("   ✌️  2 นิ้วชี้ขึ้น (ชี้,กลาง) = Scroll Left/Right")
        print("   ✊ กำมือ = หยุดทำงาน")
        print("   ❌ กด 'q' เพื่อออกจากโปรแกรม")

    def reset_timers(self):
        """รีเซ็ตเวลาทั้งหมด"""
        self.last_screenshot_time = 0
        self.last_scroll_time = 0
        self.last_horizontal_scroll_time = 0
        self.last_gesture = None

    def distance(self, p1, p2):
        """คำนวณระยะห่างระหว่างจุด 2 จุด"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def is_finger_up(self, tip, pip):
        """ตรวจสอบว่านิ้วชี้ขึ้นหรือไม่"""
        return tip.y < pip.y

    def detect_gesture(self, hand_landmarks):
        """ตรวจจับท่าทางมือและส่งคืนชนิดของท่าทาง"""
        landmarks = hand_landmarks.landmark
        
        try:
            # จุดสำคัญของนิ้วต่างๆ
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[1]
            
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            index_mcp = landmarks[5]
            
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            middle_mcp = landmarks[9]
            
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            ring_mcp = landmarks[13]
            
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            pinky_mcp = landmarks[17]
            
            # ตรวจสอบว่านิ้วไหนชี้ขึ้น
            index_up = self.is_finger_up(index_tip, index_pip)
            middle_up = self.is_finger_up(middle_tip, middle_pip)
            ring_up = self.is_finger_up(ring_tip, ring_pip)
            pinky_up = self.is_finger_up(pinky_tip, pinky_pip)
            thumb_up = thumb_tip.x < thumb_mcp.x  # นิ้วโป้งแยกต่างหาก
            
            # ตรวจสอบการกำมือ
            index_length = self.distance(index_mcp, index_tip)
            index_full_length = self.distance(index_mcp, index_pip) * 2.5
            
            middle_length = self.distance(middle_mcp, middle_tip)
            middle_full_length = self.distance(middle_mcp, middle_pip) * 2.5
            
            ring_length = self.distance(ring_mcp, ring_tip)
            ring_full_length = self.distance(ring_mcp, ring_pip) * 2.5
            
            pinky_length = self.distance(pinky_mcp, pinky_tip)
            pinky_full_length = self.distance(pinky_mcp, pinky_pip) * 2.5
            
            fist_threshold = 0.6
            is_fist = (
                index_length < index_full_length * fist_threshold and
                middle_length < middle_full_length * fist_threshold and
                ring_length < ring_full_length * fist_threshold and
                pinky_length < pinky_full_length * fist_threshold
            )
            
            if is_fist:
                return "fist", {}
            
            # ตรวจสอบ 3 นิ้วชี้ขึ้น (Screenshot)
            if index_up and middle_up and ring_up and not pinky_up:
                # ตรวจสอบว่านิ้วแยกจากกัน
                index_middle_separated = abs(index_tip.x - middle_tip.x) > 0.03
                middle_ring_separated = abs(middle_tip.x - ring_tip.x) > 0.03
                
                if index_middle_separated and middle_ring_separated:
                    return "three_fingers", {}
            
            # ตรวจสอบ 2 นิ้วชี้ขึ้น (Horizontal Scroll)
            elif index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
                distance_x = abs(index_tip.x - middle_tip.x)
                return "two_fingers", {"distance": distance_x}
            
            # ตรวจสอบ 1 นิ้วชี้ขึ้น (Vertical Scroll)
            elif index_up and not middle_up and not ring_up and not pinky_up:
                # คำนวณทิศทางของนิ้วชี้
                dx = index_tip.x - index_pip.x
                dy = index_tip.y - index_pip.y
                
                angle_rad = math.atan2(dx, -dy)
                angle_deg = math.degrees(angle_rad)
                if angle_deg < 0:
                    angle_deg += 360
                
                if angle_deg < 30 or angle_deg > 330:
                    return "scroll_up", {}
                elif 150 < angle_deg < 210:
                    return "scroll_down", {}
            
            return "none", {}
            
        except Exception as e:
            print(f"Error in gesture detection: {e}")
            return "none", {}

    def handle_screenshot(self):
        """จัดการการถ่าย Screenshot"""
        current_time = time.time()
        if current_time - self.last_screenshot_time > self.screenshot_delay:
            try:
                screenshot = pyautogui.screenshot()
                self.screenshot_count += 1
                filename = f"screenshot_{self.screenshot_count}_{int(time.time())}.png"
                screenshot.save(filename)
                print(f"📸 Screenshot saved: {filename}")
                self.last_screenshot_time = current_time
                return True
            except Exception as e:
                print(f"❌ Error taking screenshot: {e}")
        return False

    def handle_scroll(self, direction):
        """จัดการการเลื่อนแนวตั้ง"""
        current_time = time.time()
        if current_time - self.last_scroll_time > self.scroll_cooldown:
            if direction == "up":
                pyautogui.scroll(self.scroll_amount)
                print("⬆️ Scroll Up")
            elif direction == "down":
                pyautogui.scroll(-self.scroll_amount)
                print("⬇️ Scroll Down")
            
            self.last_scroll_time = current_time
            return True
        return False

    def handle_horizontal_scroll(self, distance):
        """จัดการการเลื่อนแนวนอน"""
        current_time = time.time()
        if current_time - self.last_horizontal_scroll_time > self.horizontal_scroll_cooldown:
            if distance < self.distance_threshold:
                pyautogui.press('right')
                print("➡️ Scroll Right")
                self.last_horizontal_scroll_time = current_time
                return "RIGHT"
            elif distance > self.distance_threshold:
                pyautogui.press('left')
                print("⬅️ Scroll Left")
                self.last_horizontal_scroll_time = current_time
                return "LEFT"
        return "NONE"

    def run(self):
        """เรียกใช้งานโปรแกรม"""
        screenshot_taken = False
        current_action = "WAITING"
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("❌ ไม่สามารถอ่านภาพจากกล้อง")
                break

            # กลับภาพและแปลงสี
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            # รีเซ็ตสถานะ
            screenshot_taken = False
            current_action = "WAITING"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # วาด landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # ตรวจจับท่าทาง
                    gesture_type, gesture_data = self.detect_gesture(hand_landmarks)
                    
                    # จัดการท่าทางต่างๆ
                    if gesture_type == "three_fingers":
                        screenshot_taken = self.handle_screenshot()
                        current_action = "SCREENSHOT"
                        
                    elif gesture_type == "two_fingers":
                        action = self.handle_horizontal_scroll(gesture_data["distance"])
                        current_action = f"H_SCROLL_{action}"
                        
                    elif gesture_type == "scroll_up":
                        self.handle_scroll("up")
                        current_action = "SCROLL_UP"
                        
                    elif gesture_type == "scroll_down":
                        self.handle_scroll("down")
                        current_action = "SCROLL_DOWN"
                        
                    elif gesture_type == "fist":
                        current_action = "STOP"
                        
                    else:
                        current_action = "NONE"

            # แสดงข้อมูลบนหน้าจอ
            y_offset = 30
            
            # สถานะปัจจุบัน
            color = (0, 255, 0) if current_action != "WAITING" else (255, 255, 255)
            cv2.putText(frame, f"Action: {current_action}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 30
            
            # จำนวน Screenshot
            cv2.putText(frame, f"Screenshots: {self.screenshot_count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            
            # คำแนะนำ
            cv2.putText(frame, "3 fingers = Screenshot | 1 finger = V.Scroll", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20
            cv2.putText(frame, "2 fingers = H.Scroll | Fist = Stop", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # แสดงการแจ้งเตือนการถ่าย Screenshot
            if screenshot_taken:
                cv2.putText(frame, "SCREENSHOT TAKEN!", (50, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # แสดงผล
            cv2.imshow('Combined Hand Gesture Control', frame)

            # ตรวจสอบการกดปุ่ม
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # ปิดทุกอย่าง
        self.cleanup()

    def cleanup(self):
        """ทำความสะอาดและปิดโปรแกรม"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("👋 ปิดโปรแกรมเรียบร้อย")

def main():
    controller = HandGestureController()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n🛑 ผู้ใช้หยุดโปรแกรม")
        controller.cleanup()
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        controller.cleanup()

if __name__ == "__main__":
    main()
    