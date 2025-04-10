import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 허용

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np


# 🔹 HS 히스토그램 추출 함수
def extract_hs_histogram_from_cropped(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# 🔹 히스토그램 비교 함수
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# 🔹 YOLOv8 Pose 모델 로드
model = YOLO("yolov8n-pose.pt")

# 🔹 A, B 이미지 경로 설정
image_paths = [
    "C:/Users/Ok/Desktop/rgb_00056_1.jpg",  # A
    "C:/Users/Ok/Desktop/rgb_00128_1.jpg",  # B
]

# 🔹 결과 저장 리스트
a_crops, b_crops = [], []
a_hists, b_hists = [], []

# 🔹 각 이미지 처리
for img_idx, image_path in enumerate(image_paths):
    print(f"\n=== 🔍 이미지 {img_idx} 분석 중: {image_path} ===")

    img = cv2.imread(image_path)
    if img is None:
        print("❗이미지를 불러올 수 없습니다.")
        continue

    results = model(img)
    boxes = results[0].boxes

    if boxes is None or boxes.xyxy.shape[0] == 0:
        print("❗사람이 감지되지 않았습니다.")
        continue

    for person_id, box in enumerate(boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)

        # 경계값 보정
        h, w, _ = img.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        cropped_img = img[y1:y2, x1:x2]
        if cropped_img.size == 0:
            print(f"❗크롭 실패: {x1}, {y1}, {x2}, {y2}")
            continue

        hist = extract_hs_histogram_from_cropped(cropped_img)

        if img_idx == 0:
            a_crops.append(cropped_img)
            a_hists.append(hist)
        else:
            b_crops.append(cropped_img)
            b_hists.append(hist)

# 🔹 유사도 비교 및 시각화
for i, (a_img, a_hist) in enumerate(zip(a_crops, a_hists)):
    for j, (b_img, b_hist) in enumerate(zip(b_crops, b_hists)):
        similarity = compare_histograms(a_hist, b_hist)
        print(f"\n사람 A[{i}] vs B[{j}] 유사도: {similarity:.4f}")

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
