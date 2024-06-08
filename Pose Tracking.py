import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('PoseVideos.mp4')

frame_skip = 2  # Process every 2nd frame
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    start_time = time.time()

    # Resize frame for faster processing
    img = cv2.resize(img, (720, 720))
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRBG)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (225, 0, 0), cv2.FILLED)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(img, f'FPS: {int(fps)}', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()