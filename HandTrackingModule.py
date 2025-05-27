# 트랙킹 모듈 생성성

import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Hand tracking
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw = True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks) --> Check Tracking

        # Hand tracking indication
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mphands.HAND_CONNECTIONS)
        
        return frame

    def findPosition(self, frame, handNO=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNO]

            # Logic to indicate tracking points
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm) --> Output ID and coordinate values
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    # if id == 0:
                    cv2.circle(frame, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        
        return lmList


def main():
    # Test
    
    pTime = 0
    cTime = 0

    # Test: Open Webcam
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if lmList:
            print(lmList[4])
       
        
        # Display frame time
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        # Display frames (display frame screen, variables to be displayed, location, font, font, font size, color, thickness)
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Webcam", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
