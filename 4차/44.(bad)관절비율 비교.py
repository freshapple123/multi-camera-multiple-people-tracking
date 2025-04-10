import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np


# 🔹 유클리드 거리 계산
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# 🔹 관절 비율 벡터 추출
def extract_body_ratios(kps):
    l_shoulder, r_shoulder = kps[5], kps[6]
    l_hip, r_hip = kps[11], kps[12]

    valid = all(k[2] >= 0.5 for k in [l_shoulder, r_shoulder, l_hip, r_hip])
    if not valid:
        return None

    shoulder_width = euclidean(l_shoulder[:2], r_shoulder[:2])
    hip_width = euclidean(l_hip[:2], r_hip[:2])
    shoulder_center = (np.array(l_shoulder[:2]) + np.array(r_shoulder[:2])) / 2
    hip_center = (np.array(l_hip[:2]) + np.array(r_hip[:2])) / 2
    torso_length = euclidean(shoulder_center, hip_center)

    if hip_width == 0 or torso_length == 0:
        return None

    return np.array([
        shoulder_width / hip_width,
        shoulder_width / torso_length,
        hip_width / torso_length
    ])


# 🔹 관절 비율 유사도 비교
def compare_ratios(r1, r2):
    if r1 is None or r2 is None:
        return 0.0
    return 1 - np.linalg.norm(r1 - r2)


# 🔹 모델 로드
model = YOLO("yolov8n-pose.pt")

# 🔹 이미지 경로 설정
image_paths = [
    "C:/Users/Ok/Desktop/rgb_00056_1.jpg",  # A
    "C:/Users/Ok/Desktop/rgb_00128_1.jpg",  # B
]

# 🔹 결과 저장
images = []
ratios_list = []
person_images = []

# 🔹 이미지별 처리
for img_path in image_paths:
    print(f"\n📷 이미지 분석 중: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("❗이미지 로드 실패")
        continue

    results = model(img)
    keypoints = results[0].keypoints

    if keypoints is None or keypoints.data.shape[0] == 0:
        print("❗사람 감지 실패")
        continue

    kp_array = keypoints.data.cpu().numpy()
    num_people = kp_array.shape[0]

    for person_id in range(num_people):
        kps = kp_array[person_id]
        ratio = extract_body_ratios(kps)

        h, w, _ = img.shape
        person_img = img.copy()

        # 박스가 있다면 해당 영역 자르기 (선택 사항)
        box = results[0].boxes.xyxy
        if box is not None and len(box) > person_id:
            x1, y1, x2, y2 = box[person_id]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            person_img = img[y1:y2, x1:x2]

        person_images.append(person_img)
        ratios_list.append(ratio)

# 🔹 유사도 비교 및 출력
print("\n🔍 관절 비율 유사도 비교 결과:")
for i in range(len(person_images)):
    for j in range(len(person_images)):
        if i >= j:
            continue

        sim = compare_ratios(ratios_list[i], ratios_list[j])
        print(f"   🔸 사람 {i} vs 사람 {j} → 관절 유사도: {sim:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(cv2.cvtColor(person_images[i], cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"사람 {i}")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(person_images[j], cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"사람 {j}")
        axes[1].axis("off")

        plt.suptitle(f"관절 비율 유사도: {sim:.4f}")
        plt.tight_layout()
        plt.show()
