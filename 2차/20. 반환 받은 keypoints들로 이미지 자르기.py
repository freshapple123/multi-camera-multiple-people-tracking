import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 허용

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# YOLOv8 Pose 모델 로드
model = YOLO("yolov8n-pose.pt")

# 정지영상 로드
image_path = "C:/Users/Ok/Desktop/5th_angle/person_1.jpg"
img = cv2.imread(image_path)

# YOLOv8 Pose 추론
results = model(img)

# 첫 번째 사람의 키포인트 정보 가져오기
keypoints = results[0].keypoints
kp_array = keypoints.data.cpu().numpy()

# 필요한 키포인트 인덱스: [왼쪽 어깨(5), 오른쪽 어깨(6), 왼쪽 골반(11), 오른쪽 골반(12)]
desired_indices = [5, 6, 11, 12]
selected_points = kp_array[0][desired_indices]

# 각 키포인트에서 (x, y)만 추출
xy_points = selected_points[:, :2]

# 사각형 영역의 최소 x, y와 최대 x, y 계산
x_min = int(np.min(xy_points[:, 0]))
x_max = int(np.max(xy_points[:, 0]))
y_min = int(np.min(xy_points[:, 1]))
y_max = int(np.max(xy_points[:, 1]))

# 이미지 경계 벗어나지 않도록 조정
h, w, _ = img.shape
x_min = max(x_min, 0)
x_max = min(x_max, w)
y_min = max(y_min, 0)
y_max = min(y_max, h)

# 사각형 영역 잘라내기 (crop)
cropped_img = img[y_min:y_max, x_min:x_max]

# 결과 출력
labels = ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"]
for label, point in zip(labels, selected_points):
    x, y, conf = point
    print(f"{label}: x={x:.1f}, y={y:.1f}, confidence={conf:.2f}")

# 잘라낸 이미지 보기 (matplotlib 이용)
plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
plt.title("Cropped Torso")
plt.axis("off")
plt.show()
