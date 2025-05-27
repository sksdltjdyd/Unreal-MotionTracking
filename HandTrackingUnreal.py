import cv2
import mediapipe as mp
import time
import HandTrackingModuleMultiHands as htm
from pythonosc import udp_client
import threading
from collections import deque

class HandTracker:
    def __init__(self, osc_ip="127.0.0.1", osc_port=8000):
        # OSC Client Settings
        self.client = udp_client.SimpleUDPClient(osc_ip, osc_port)
        
        # Finger endpoint index (thumb, index finger, middle finger, ring finger, pinky)
        self.fingertip_ids = [4, 8, 12, 16, 20]
        
        # Finger Name Mapping
        self.finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        
        # Hand Sensor (Enable both hand detection)
        self.detector = htm.HandDetector(maxHands=2)
        
        # Webcam Settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # For FPS calculation
        self.pTime = 0
        
        # Buffer for data smoothing
        self.smoothing_buffer_size = 3
        self.left_hand_buffer = {finger: deque(maxlen=self.smoothing_buffer_size) 
                                for finger in self.finger_names}
        self.right_hand_buffer = {finger: deque(maxlen=self.smoothing_buffer_size) 
                                 for finger in self.finger_names}
        
        print("HandTracker 초기화 완료")
        print(f"OSC 통신: {osc_ip}:{osc_port}")
        print("지원 기능: 양손 동시 추적, 모든 손가락 끝점 전송")
    
    def smooth_coordinates(self, buffer, new_coords):
        """좌표 스무딩을 위한 함수"""
        buffer.append(new_coords)
        if len(buffer) > 0:
            avg_x = sum(coord[0] for coord in buffer) / len(buffer)
            avg_y = sum(coord[1] for coord in buffer) / len(buffer)
            return [avg_x, avg_y]
        return new_coords
    
    def send_hand_data(self, hand_landmarks, hand_label):
        """각 손의 모든 손가락 끝점 데이터를 OSC로 전송"""
        try:
            # Extract and transfer coordinates for each finger endpoint
            for i, finger_id in enumerate(self.fingertip_ids):
                if finger_id < len(hand_landmarks):
                    x, y = hand_landmarks[finger_id][1], hand_landmarks[finger_id][2]
                    finger_name = self.finger_names[i]
                    
                    # Coordinate Normalization (with a range of 0-1)
                    normalized_x = x / 1280.0  # 웹캠 너비로 나누기
                    normalized_y = y / 720.0   # 웹캠 높이로 나누기
                    
                    # Apply smoothing
                    if hand_label == "Left":
                        smoothed_coords = self.smooth_coordinates(
                            self.left_hand_buffer[finger_name], 
                            [normalized_x, normalized_y]
                        )
                    else:
                        smoothed_coords = self.smooth_coordinates(
                            self.right_hand_buffer[finger_name], 
                            [normalized_x, normalized_y]
                        )
                    
                    # Send OSC Messages
                    osc_address = f"/{hand_label.lower()}_hand/{finger_name}"
                    self.client.send_message(osc_address, smoothed_coords)
                    
                    # Debug Output (uncomment if required)
                    # print(f"Sent: {osc_address} [{smoothed_coords[0]:.3f}, {smoothed_coords[1]:.3f}]")
            
            # Hand-wide center plot transmission (based on wrist)
            if len(hand_landmarks) > 0:
                wrist_x, wrist_y = hand_landmarks[0][1], hand_landmarks[0][2]
                normalized_wrist = [wrist_x / 1280.0, wrist_y / 720.0]
                wrist_address = f"/{hand_label.lower()}_hand/wrist"
                self.client.send_message(wrist_address, normalized_wrist)
                
        except Exception as e:
            print(f"OSC 전송 오류 ({hand_label}): {e}")
    
    def run(self):
        """메인 실행 루프"""
        print("손 추적 시작... 'q'를 눌러 종료")
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break
            
            # Frame Inversion (mirror effect)
            frame = cv2.flip(frame, 1)
            
            # Hand Sensing
            frame = self.detector.findHands(frame, draw=True)
            
            # Multiple hand position detection
            multi_hand_landmarks = self.detector.findMultiHandPositions(frame)
            
            if multi_hand_landmarks:
                for hand_info in multi_hand_landmarks:
                    hand_landmarks = hand_info['landmarks']
                    hand_label = hand_info['label']  # 'Left' OR 'Right'
                    
                    # Transfer data from each hand to OSC
                    self.send_hand_data(hand_landmarks, hand_label)
            
            # Calculate and display FPS
            cTime = time.time()
            fps = 1 / (cTime - self.pTime) if self.pTime != 0 else 0
            self.pTime = cTime
            
            # Print Information
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Hands: {len(multi_hand_landmarks) if multi_hand_landmarks else 0}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Displaying the screen
            cv2.imshow("Hand Tracking - OSC Communication", frame)
            
            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Transmission speed control
            time.sleep(0.02)  # 50Hz Update
        
        # Return
        self.cap.release()
        cv2.destroyAllWindows()
        print("손 추적 종료")

# Methods to add to HandTrackingModule.py
class HandDetectorExtended(htm.HandDetector):
    def findMultiHandPositions(self, img, draw=True):
        """다중 손 위치 감지 (양손 지원)"""
        multiHandData = []
        
        if self.results.multi_hand_landmarks:
            for hand_idx, (handLms, handedness) in enumerate(
                zip(self.results.multi_hand_landmarks, self.results.multi_handedness)
            ):
                myHand = []
                hand_label = handedness.classification[0].label
                
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    myHand.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                multiHandData.append({
                    'landmarks': myHand,
                    'label': hand_label
                })
        
        return multiHandData

if __name__ == "__main__":
    # Examples
    try:
        tracker = HandTracker(osc_ip="127.0.0.1", osc_port=8000)
        tracker.run()
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")