import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 허용

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# YOLOv8 Pose 모델 로드
model = YOLO("yolov8n-pose.pt")

# 정지영상 로드
# image_path = "C:/Users/Ok/Desktop/1st_angle/person_2.jpg"
image_path = "C:/Users/Ok/Desktop/5th_angle/person_2.jpg"
img = cv2.imread(image_path)

# YOLOv8 Pose 추론
results = model(img)

# 첫 번째 사람의 키포인트 정보 가져오기
keypoints = results[0].keypoints
kp_array = keypoints.data.cpu().numpy()

# 필요한 키포인트 인덱스와 라벨
desired_indices = [5, 6, 11, 12]
labels = ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"]

# 유효한 점만 추출 (신뢰도 0.5 이상)
valid_points = []
valid_labels = []

for idx, label in zip(desired_indices, labels):
    x, y, conf = kp_array[0][idx]
    if conf >= 0.5:
        valid_points.append([x, y])
        valid_labels.append(label)
    else:
        print(f"[무시됨] {label}: 낮은 신뢰도 (confidence={conf:.2f})")

valid_points = np.array(valid_points)

# 유효한 점이 3개 이상일 경우에만 사각형 추출
if len(valid_points) >= 3:
    # 좌표의 최소/최대값으로 박스 계산
    x_min = int(np.min(valid_points[:, 0]))
    x_max = int(np.max(valid_points[:, 0]))
    y_min = int(np.min(valid_points[:, 1]))
    y_max = int(np.max(valid_points[:, 1]))

    # 이미지 크기 안으로 제한
    h, w, _ = img.shape
    x_min = max(x_min, 0)
    x_max = min(x_max, w)
    y_min = max(y_min, 0)
    y_max = min(y_max, h)

    # 잘라내기
    cropped_img = img[y_min:y_max, x_min:x_max]

    # 좌표 출력
    for label, point in zip(valid_labels, valid_points):
        print(f"{label}: x={point[0]:.1f}, y={point[1]:.1f}")

    # 잘라낸 이미지 보기
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Torso (filtered)")
    plt.axis("off")
    plt.show()
else:
    print(
        "\n❗신뢰도가 0.5 이상인 키포인트가 3개 미만입니다. 사각형을 만들 수 없습니다."
    )
