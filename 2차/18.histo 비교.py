import cv2
import numpy as np


def extract_color_histogram(image):
    """전체 이미지에서 색상 히스토그램 추출"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def compare_histograms(hist1, hist2):
    """두 히스토그램 간의 유사도 계산 (코렐레이션)"""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# 이미지 불러오기
frame1 = cv2.imread("C:/Users/Ok/Desktop/1st_angle/person_2.jpg")
frame2 = cv2.imread("C:/Users/Ok/Desktop/5th_angle/person_2.jpg")

# 이미지 로드 성공 확인
if frame1 is None or frame2 is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
else:
    # 히스토그램 추출 및 비교
    hist1 = extract_color_histogram(frame1)
    hist2 = extract_color_histogram(frame2)
    similarity = compare_histograms(hist1, hist2)

    print(f"Color Similarity: {similarity}")
