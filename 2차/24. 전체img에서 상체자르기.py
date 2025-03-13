import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 허용

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

# 각 사람에 대해 키포인트 및 잘라낸 이미지 출력
for person_idx, person_kp in enumerate(kp_array):
    print(f"\nPerson {person_idx + 1}")
    selected_points = person_kp[desired_indices]

    # 키포인트 출력
    for label, point in zip(labels, selected_points):
        x, y, conf = point
        print(f"{label}: x={x:.1f}, y={y:.1f}, confidence={conf:.2f}")

    # (x, y) 좌표만 추출
    xy_points = selected_points[:, :2]

    # 사각형 영역 계산
    x_min = int(np.min(xy_points[:, 0]))
    x_max = int(np.max(xy_points[:, 0]))
    y_min = int(np.min(xy_points[:, 1]))
    y_max = int(np.max(xy_points[:, 1]))

    # 이미지 경계 벗어나지 않도록 보정
    h, w, _ = img.shape
    x_min = max(x_min, 0)
    x_max = min(x_max, w)
    y_min = max(y_min, 0)
    y_max = min(y_max, h)

    # 사각형 영역 잘라내기
    cropped_img = img[y_min:y_max, x_min:x_max]

    # 결과 출력
    plt.figure()
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Cropped Torso - Person {person_idx + 1}")
    plt.axis("off")
    plt.show()

    # 이미지 저장 (선택 사항)
    # save_path = f"C:/Users/Ok/Desktop/cropped_person_{person_idx + 1}.jpg"
    # cv2.imwrite(save_path, cropped_img)
