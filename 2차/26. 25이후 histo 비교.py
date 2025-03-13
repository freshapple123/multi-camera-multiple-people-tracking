import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 허용

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# YOLOv8 Pose 모델 로드
model = YOLO("yolov8n-pose.pt")

# 이미지 경로 리스트
image_paths = [
    "C:/Users/Ok/Desktop/rgb_00000_5.jpg",
    "C:/Users/Ok/Desktop/rgb_00000_1.jpg",  # 여기에 두 번째 이미지 경로
]

# 키포인트 인덱스 및 라벨
desired_indices = [5, 6, 11, 12]  # 왼쪽/오른쪽 어깨, 왼쪽/오른쪽 골반
labels = ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"]

# 이미지마다 반복
for img_idx, image_path in enumerate(image_paths):
    print(f"\n=== 🔍 이미지 {img_idx} 분석 중: {image_path} ===")

    img = cv2.imread(image_path)
    if img is None:
        print("❗이미지를 불러올 수 없습니다.")
        continue

    results = model(img)
    keypoints = results[0].keypoints

    if keypoints is None or keypoints.data.shape[0] == 0:
        print("❗사람이 감지되지 않았습니다.")
        continue

    kp_array = keypoints.data.cpu().numpy()
    num_people = kp_array.shape[0]

    for person_id in range(num_people):
        valid_points = []
        valid_labels = []

        for idx, label in zip(desired_indices, labels):
            x, y, conf = kp_array[person_id][idx]
            if conf >= 0.5:
                valid_points.append([x, y])
                valid_labels.append(label)
            else:
                print(
                    f"[무시됨 - 이미지 {img_idx}, 사람 {person_id}] {label}: 낮은 신뢰도 (confidence={conf:.2f})"
                )

        valid_points = np.array(valid_points)

        if len(valid_points) >= 3:
            x_min = int(np.min(valid_points[:, 0]))
            x_max = int(np.max(valid_points[:, 0]))
            y_min = int(np.min(valid_points[:, 1]))
            y_max = int(np.max(valid_points[:, 1]))

            h, w, _ = img.shape
            x_min = max(x_min, 0)
            x_max = min(x_max, w)
            y_min = max(y_min, 0)
            y_max = min(y_max, h)

            cropped_img = img[y_min:y_max, x_min:x_max]

            print(f"\n✅ 이미지 {img_idx} - 사람 {person_id} 유효한 키포인트:")
            for label, point in zip(valid_labels, valid_points):
                print(f"{label}: x={point[0]:.1f}, y={point[1]:.1f}")

            plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Image {img_idx} - Person {person_id}")
            plt.axis("off")
            plt.show()

            # 저장 예시 (원하면 사용)
            # save_path = f"C:/Users/Ok/Desktop/cropped_img{img_idx}_person{person_id}.jpg"
            # cv2.imwrite(save_path, cropped_img)

        else:
            print(
                f"\n❗이미지 {img_idx} - 사람 {person_id}: 신뢰도 0.5 이상 키포인트가 3개 미만입니다."
            )
