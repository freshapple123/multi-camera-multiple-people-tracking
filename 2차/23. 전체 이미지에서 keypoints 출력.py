import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# YOLOv8 Pose 모델 로드
model = YOLO("yolov8n-pose.pt")

# 이미지 경로 설정
image_path = "C:/Users/Ok/Desktop/rgb_00000_5.jpg"
img = cv2.imread(image_path)

# YOLOv8 Pose 추론
results = model(img)

# 키포인트 데이터 가져오기
keypoints = results[0].keypoints
kp_array = keypoints.data.cpu().numpy()  # [num_people, num_keypoints, 3]

# 관심 있는 키포인트 인덱스 (왼쪽 어깨, 오른쪽 어깨, 왼쪽 골반, 오른쪽 골반)
desired_indices = [5, 6, 11, 12]
labels = ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"]

# 각 사람에 대해 키포인트 출력
for person_idx, person_kp in enumerate(kp_array):
    print(f"\nPerson {person_idx + 1}")
    selected_points = person_kp[desired_indices]
    
    for label, point in zip(labels, selected_points):
        x, y, conf = point
        print(f"{label}: x={x:.1f}, y={y:.1f}, confidence={conf:.2f}")
