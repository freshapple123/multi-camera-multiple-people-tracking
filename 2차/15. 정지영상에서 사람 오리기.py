import cv2
import torch
from ultralytics import YOLO

# YOLOv8 모델 로드 (사전 학습된 모델 사용)
model = YOLO("yolov8n.pt")  # 경량 모델 사용 (nano)

# 이미지 로드
image_path = "rgb_00000_6.jpg"  # 입력 이미지 경로
image = cv2.imread(image_path)

# 객체 탐지 수행
results = model(image)

# 저장할 인물 번호
person_count = 0

# 탐지 결과 확인
for result in results:
    boxes = result.boxes  # Bounding boxes 정보 가져오기
    for box in boxes:
        cls = int(box.cls[0])  # 객체 클래스 (0: 사람)
        if cls == 0:  # 사람만 필터링
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # BBox 좌표

            # 이미지 자르기
            cropped_person = image[y1:y2, x1:x2]

            # 자른 이미지 저장
            output_path = f"person_{person_count}.jpg"
            cv2.imwrite(output_path, cropped_person)
            print(f"Saved: {output_path}")

            person_count += 1  # 저장된 인물 수 증가
