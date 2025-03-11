import cv2
import numpy as np


def extract_hs_histogram(image, bbox):
    """특정 영역에서 Hue(색상) + Saturation(채도) 히스토그램 추출"""
    x, y, w, h = bbox
    roi = image[y : y + h, x : x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1], None, [50, 60], [0, 180, 0, 256]
    )  # Hue: 50 bins, Saturation: 60 bins
    cv2.normalize(hist, hist)

    return hist.flatten()  # 2D 히스토그램을 벡터 형태로 변환


def compare_histograms(hist1, hist2):
    """두 히스토그램 간의 유사도 계산 (코렐레이션)"""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


hist1 = extract_hs_histogram(frame1, bbox1)
hist2 = extract_hs_histogram(frame2, bbox2)
similarity = compare_histograms(hist1, hist2)

print(f"HS Color Similarity: {similarity}")
