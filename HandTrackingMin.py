import cv2
import mediapipe as mp
import time

# 테스트: 웹캠 열기
cap = cv2.VideoCapture(0)

#손 트랙킹
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks) --> 트랙킹 확인

    #손 트랙킹 표시
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm) --> 아이디와 좌표값 출력
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
               #print(id, cx, cy) --> 아이디와 노멀라이즈 좌표값 출력
                if id == 0:
                    cv2.circle(frame, (cx,cy), 15, (255,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(frame, handLms, mphands.HAND_CONNECTIONS)


    #프레임 타임 표시
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #프레임 표시(프레임 화면, 표시할 변수, 위치, 폰트, 글씨 크기, 색, 굵기 표시)
    cv2.putText(frame,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,255), 3)

    cv2.imshow("Webcam", frame)
    cv2.waitKey(1)