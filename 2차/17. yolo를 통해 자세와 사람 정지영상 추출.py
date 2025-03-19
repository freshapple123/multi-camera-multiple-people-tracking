import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# YOLOv8 Pose 모델 로드
model = YOLO("yolov8n-pose.pt")  # 모델 경로

# 정지영상 로드
# image_path = "C:/Users/Ok/Desktop/1st_angle/person_0.jpg"
# image_path = "C:/Users/Ok/Desktop/1st_angle/person_0.jpg"
# image_path = "C:/Users/Ok/Desktop/5th_angle/person_1.jpg"
image_path = "C:/Users/Ok/Desktop/rgb_00391_3.jpg"
img = cv2.imread(image_path)

# 이미지에서 포즈 추적
results = model(img)  # YOLOv8 Pose 추론

# 결과 출력 (추적된 사람과 포즈)
results[0].show()  # 첫 번째 결과에 대해 show() 호출

# 포즈 정보 가져오기
keypoints = results[0].keypoints  # 첫 번째 결과에서 키포인트 정보 (x, y 좌표)
print("Keypoints for each person:")

# 각 사람의 키포인트 좌표 출력
for person_idx, person_keypoints in enumerate(keypoints):
    print(f"Person {person_idx + 1}:")
    for i, point in enumerate(person_keypoints):
        x, y, conf = point  # x, y 좌표 및 신뢰도
        print(f"  Point {i}: x = {x}, y = {y}, confidence = {conf}")

# 키포인트 좌표를 이미지에 그리기
for person_keypoints in keypoints:
    for i, point in enumerate(person_keypoints):
        x, y, conf = point  # x, y 좌표 및 신뢰도
        cv2.circle(
            img, (int(x), int(y)), 5, (0, 255, 0), -1
        )  # 각 키포인트를 이미지에 표시

# 결과 이미지 출력
cv2.imshow("Pose Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
