# 양손 트랙킹 테스트

import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, 
            max_num_hands=self.maxHands, 
            min_detection_confidence=self.detectionCon, 
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        # Set results to None during initialization
        self.results = None

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # self.mpHands.process --> self.hands.process
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        # One hand Detecting (Maintain existing compatibility)
        lmList = []
        if self.results and self.results.multi_hand_landmarks:  # self.results Add Existence Verification
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def findMultiHandPositions(self, frame, draw=True):
        # Multi-hand position detection (both hands support)
        multiHandData = []
        
        # self.results와 multi_handedness Make sure all exist
        if (self.results and 
            self.results.multi_hand_landmarks and 
            self.results.multi_handedness):
            
            for hand_idx, (handLms, handedness) in enumerate(
                zip(self.results.multi_hand_landmarks, self.results.multi_handedness)
            ):
                myHand = []
                hand_label = handedness.classification[0].label
                
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    myHand.append([id, cx, cy])
                    
                    if draw:
                        # Use different colors depending on your hand
                        color = (255, 0, 0) if hand_label == "Left" else (0, 0, 255)
                        cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)
                
                multiHandData.append({
                    'landmarks': myHand,
                    'label': hand_label,
                    'index': hand_idx
                })
        
        return multiHandData

    def getHandLabel(self, handNo=0):
        # Import labels for specific hands (Left/Right)
        if self.results and self.results.multi_handedness:  # self.results Add Existence Verification
            if handNo < len(self.results.multi_handedness):
                return self.results.multi_handedness[handNo].classification[0].label
        return None

def main():
    # Test Def
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    
    # Make sure your webcam is open properly
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 다른 프로그램에서 사용 중인지 확인하세요.")
        return
    
    detector = HandDetector()
    
    print("손 추적 시작... 'q'를 눌러 종료")
    
    while True:
        success, frame = cap.read()
        
        # Confirm frame read failure
        if not success:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break
        
        # Frame Inversion (mirror effect)
        frame = cv2.flip(frame, 1)
        
        frame = detector.findHands(frame)
        
        # Multi-hand position detection test
        multiHandData = detector.findMultiHandPositions(frame)
        
        if multiHandData:
            for hand_info in multiHandData:
                hand_landmarks = hand_info['landmarks']
                hand_label = hand_info['label']
                print(f"{hand_label} Hand detected with {len(hand_landmarks)} landmarks")
        
        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime

        # Display FPS and the number of hands
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Hands: {len(multiHandData)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Hand Tracking Test", frame)
        
        # Press the 'q' key to end
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Return
    cap.release()
    cv2.destroyAllWindows()
    print("손 추적 종료")

if __name__ == "__main__":
    main()