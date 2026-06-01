import cv2
import mediapipe as mp
import csv

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

cap = cv2.VideoCapture("D:/pose_project/fall.mp4")

# 👉 저장할 파일
file = open('D:/pose_project/fall_data.csv', mode='w', newline='')
writer = csv.writer(file)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        coords = []
        for lm in results.pose_landmarks.landmark:
            coords.append(lm.x)
            coords.append(lm.y)

        # 👉 CSV 저장
        writer.writerow(coords)

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    frame = cv2.resize(frame, (800, 600))
    cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()