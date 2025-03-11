import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기
# image_path = "C:/Users/Ok/Desktop/a-person/b-person/pants_0.png"
image_path = "C:/Users/Ok/Desktop/3_b_pants.png"
# 업로드된 이미지 경로
image = cv2.imread(image_path)

# 이미지를 BGR에서 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 마스크 생성: 검은색 배경 제외 (픽셀 값이 (0, 0, 0)인 영역)
mask = cv2.inRange(image_rgb, (1, 1, 1), (255, 255, 255))  # (0,0,0)이 아닌 영역만 포함

# 히스토그램 계산 (마스크 적용)
colors = ("r", "g", "b")  # 컬러 채널
histograms = {}
plt.figure(figsize=(8, 6))

for i, color in enumerate(colors):
    hist = cv2.calcHist([image_rgb], [i], mask, [256], [0, 256])  # 마스크 적용
    histograms[color] = hist
    plt.plot(hist, color=color, label=f"{color.upper()} channel")

# 히스토그램 출력
plt.title("Color Histogram for Pants (Background Removed)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()
