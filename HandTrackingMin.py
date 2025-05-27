import cv2
import mediapipe as mp
import time

# Test : Opne Webcam
cap = cv2.VideoCapture(0)

# HandTracking
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks) --> Check Tracking

    #손 트랙킹 표시
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm) --> Return ID, Axis
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
               #print(id, cx, cy) --> Return ID, Normalized Axis
                if id == 0:
                    cv2.circle(frame, (cx,cy), 15, (255,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(frame, handLms, mphands.HAND_CONNECTIONS)


    # Return Frame
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # Frame on the Screen(Fram Display, Var, Loc, Font, Size, Colour, Thickness)
    cv2.putText(frame,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,255), 3)

    cv2.imshow("Webcam", frame)
    cv2.waitKey(1)