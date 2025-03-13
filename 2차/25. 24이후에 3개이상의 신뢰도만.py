import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 허용

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# YOLOv8 Pose 모델 로드
model = YOLO("yolov8n-pose.pt")

# 이미지 경로
image_path = "C:/Users/Ok/Desktop/rgb_00000_5.jpg"
img = cv2.imread(image_path)

# YOLOv8 Pose 추론
results = model(img)

# 키포인트 추출
keypoints = results[0].keypoints
if keypoints is None or keypoints.data.shape[0] == 0:
    print("❗사람이 감지되지 않았습니다.")
else:
    kp_array = keypoints.data.cpu().numpy()
    num_people = kp_array.shape[0]

    desired_indices = [5, 6, 11, 12]
    labels = ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"]

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
                    f"[무시됨 - 사람 {person_id}] {label}: 낮은 신뢰도 (confidence={conf:.2f})"
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

            print(f"\n✅ 사람 {person_id} - 유효한 키포인트:")
            for label, point in zip(valid_labels, valid_points):
                print(f"{label}: x={point[0]:.1f}, y={point[1]:.1f}")

            plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Person {person_id}: Cropped Torso")
            plt.axis("off")
            plt.show()

            # 저장 예시 (주석 해제 시 사용 가능)
            # save_path = f"C:/Users/Ok/Desktop/cropped_person_{person_id}.jpg"
            # cv2.imwrite(save_path, cropped_img)

        else:
            print(f"\n❗사람 {person_id}: 신뢰도 0.5 이상인 키포인트가 3개 미만입니다.")
