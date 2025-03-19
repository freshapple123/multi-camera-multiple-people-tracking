import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 허용

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np


# 🔹 HS 히스토그램 추출 함수 (cropped_img 넘기기)
def extract_hs_histogram_from_cropped(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# 🔹 히스토그램 비교 함수 (Bhattacharyya 거리 사용)
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


# 🔹 YOLOv8 Pose 모델 로드
model = YOLO("yolov8n-pose.pt")

# 🔹 A, B 이미지 경로 설정
image_paths = [
    "C:/Users/Ok/Desktop/rgb_00000_1.jpg",  # A
    "C:/Users/Ok/Desktop/rgb_00000_5.jpg",  # B
]

# 🔹 키포인트 인덱스 및 라벨
desired_indices = [5, 6, 11, 12]  # 왼쪽/오른쪽 어깨, 왼쪽/오른쪽 골반

# 🔹 결과 저장 리스트
a_crops, b_crops = [], []
a_hists, b_hists = [], []

# 🔹 각 이미지 처리 (A, B)
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

        for idx in desired_indices:
            x, y, conf = kp_array[person_id][idx]
            if conf >= 0.5:
                valid_points.append([x, y])

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

            # 🔹 수정된 히스토그램 추출
            hist = extract_hs_histogram_from_cropped(cropped_img)

            if img_idx == 0:  # A 이미지
                a_crops.append(cropped_img)
                a_hists.append(hist)
            else:  # B 이미지
                b_crops.append(cropped_img)
                b_hists.append(hist)
        else:
            print(f"❗이미지 {img_idx} - 사람 {person_id}: 유효한 키포인트 부족")

# 🔹 유사도 비교 및 시각화
for i, (a_img, a_hist) in enumerate(zip(a_crops, a_hists)):
    for j, (b_img, b_hist) in enumerate(zip(b_crops, b_hists)):
        similarity = compare_histograms(a_hist, b_hist)
        print(f"\n사람 A[{i}] vs B[{j}] 유사도: {similarity:.4f}")

        # 크롭 이미지 시각화
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"A[{i}]")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"B[{j}]")
        axes[1].axis("off")

        plt.suptitle(f"Similarity: {similarity:.4f}")
        plt.tight_layout()
        plt.show()
