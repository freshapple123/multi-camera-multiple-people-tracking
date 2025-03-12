import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# YOLOv8 Pose 모델 로드
model = YOLO("yolov8n-pose.pt")

# 정지영상 로드
image_path = "C:/Users/Ok/Desktop/5th_angle/person_1.jpg"
# image_path = "C:/Users/Ok/Desktop/1st_angle/person_0.jpg"
img = cv2.imread(image_path)

# YOLOv8 Pose 추론
results = model(img)

# 첫 번째 사람의 키포인트 정보 가져오기
keypoints = results[0].keypoints

# numpy 배열로 변환 (형태: [num_people, num_keypoints, 3])
kp_array = keypoints.data.cpu().numpy()

# 첫 번째 사람의 keypoint 좌표에서 원하는 인덱스만 추출
# 순서: [왼쪽 어깨(5), 오른쪽 어깨(6), 왼쪽 골반(11), 오른쪽 골반(12)]
desired_indices = [5, 6, 11, 12]
selected_points = kp_array[0][desired_indices]

# 결과 출력
labels = ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"]
for label, point in zip(labels, selected_points):
    x, y, conf = point
    print(f"{label}: x={x:.1f}, y={y:.1f}, confidence={conf:.2f}")
