from screeninfo import get_monitors

import time

import cv2
import cvzone.HandTrackingModule, cvzone.FaceMeshModule, cvzone.PoseModule

detector_hands = cvzone.HandTrackingModule.HandDetector(detectionCon=0.8, maxHands=2)
detector_face = cvzone.FaceMeshModule.FaceMeshDetector(maxFaces=1)
detector_pose = cvzone.PoseModule.PoseDetector()


pTime = 0

# Obter informações sobre cada monitor
""" monitors = get_monitors()

for monitor in monitors:
    print(f"Monitor {monitor.name}")
    print(f"Largura: {monitor.width} pixels")
    print(f"Altura: {monitor.height} pixels")
    print(f"Posição (X, Y): ({monitor.x}, {monitor.y})")
    print() """


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hands, frame = detector_hands.findHands(frame)
    frame, face = detector_face.findFaceMesh(frame)
    frame = detector_pose.findPose(frame)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    print(cTime)
    cv2.putText(frame, f'FPS: {str(int(fps))}', (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
