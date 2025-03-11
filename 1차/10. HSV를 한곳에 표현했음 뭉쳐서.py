import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기
# image_path = "C:/Users/Ok/Desktop/lena.jpg"  # 분석할 이미지 경로
image_path = "C:/Users/Ok/Desktop/a-person/b-person/1_pants.png"  # 분석할 이미지 경로
image = cv2.imread(image_path)
if image is None:
    print("이미지를 읽을 수 없습니다.")
    exit()

# BGR 이미지를 HSV로 변환
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 검은색 픽셀 마스크 생성 (RGB 값이 (0, 0, 0)인 경우)
mask = cv2.inRange(image, (1, 1, 1), (255, 255, 255))  # 검은색을 제외한 마스크

# HSV 채널 분리
h, s, v = cv2.split(hsv_image)

# 히스토그램 계산 (마스크 적용)
hist_h = cv2.calcHist([h], [0], mask, [256], [0, 256])
hist_s = cv2.calcHist([s], [0], mask, [256], [0, 256])
hist_v = cv2.calcHist([v], [0], mask, [256], [0, 256])

# 히스토그램 시각화 (한 그래프에 겹쳐서 표시)
plt.figure(figsize=(10, 6))

# Hue 히스토그램 (Red)
plt.plot(hist_h, color="r", label="Hue")

# Saturation 히스토그램 (Green)
plt.plot(hist_s, color="g", label="Saturation")

# Value 히스토그램 (Blue)
plt.plot(hist_v, color="b", label="Value")

plt.title("HSV Histogram (No Black)")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()
