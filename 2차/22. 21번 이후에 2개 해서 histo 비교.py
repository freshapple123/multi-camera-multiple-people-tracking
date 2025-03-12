import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# 📌 히스토그램 추출 함수
def extract_color_histogram(image):
    """HSV 색공간에서 색상 히스토그램 추출"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# 📌 히스토그램 비교 함수
def compare_histograms(hist1, hist2):
    """히스토그램 간 유사도 계산 (상관계수 기반)"""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# YOLO 모델 로드
model = YOLO("yolov8n-pose.pt")

# 처리할 이미지 경로들
image_paths = [
    "C:/Users/Ok/Desktop/1st_angle/person_2.jpg",
    "C:/Users/Ok/Desktop/5th_angle/person_2.jpg"
]

cropped_images = []

# 이미지별 처리
for image_path in image_paths:
    print(f"\n=== 처리 중: {image_path} ===")
    
    img = cv2.imread(image_path)
    results = model(img)
    keypoints = results[0].keypoints
    kp_array = keypoints.data.cpu().numpy()

    desired_indices = [5, 6, 11, 12]
    labels = ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"]

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
        cropped_images.append(cropped_img)

        for label, point in zip(valid_labels, valid_points):
            print(f"{label}: x={point[0]:.1f}, y={point[1]:.1f}")

        plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Cropped Torso - {os.path.basename(image_path)}")
        plt.axis("off")
        plt.show()
    else:
        print("❗신뢰도 0.5 이상인 키포인트가 3개 미만입니다. 사각형을 만들 수 없습니다.")

# 히스토그램 유사도 비교
if len(cropped_images) == 2:
    hist1 = extract_color_histogram(cropped_images[0])
    hist2 = extract_color_histogram(cropped_images[1])
    similarity = compare_histograms(hist1, hist2)
    print(f"\n🎨 Color Similarity (Correlation): {similarity:.4f}")
else:
    print("\n❗두 이미지 모두에서 유효한 토르소가 추출되지 않았습니다.")
