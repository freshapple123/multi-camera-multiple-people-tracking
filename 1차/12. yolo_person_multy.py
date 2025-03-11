from ultralytics import YOLO
import cv2

# YOLO 모델 불러오기 (사전 학습된 가중치 사용)
model = YOLO("yolo11n.pt")  # YOLOv8 경량 모델 사용, 필요 시 다른 모델 선택 가능

# 이미지 경로와 파일 이름 패턴
base_path = "D:/RE-id/data/retail/MMPTracking_training/retail_0"
file_pattern = "rgb_00000_{}.jpg"

# 처리할 이미지 범위
image_range = range(1, 7)  # 1에서 6까지

# 이미지 순차 처리
for i in image_range:
    image_path = f"{base_path}/{file_pattern.format(i)}"
    print(f"Processing: {image_path}")

    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        continue

    # YOLO 모델로 이미지 예측
    results = model(image)

    # 탐지된 객체 정보 가져오기
    detections = results[0].boxes.data.cpu().numpy()
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        if int(class_id) == 0:  # '0'은 COCO 데이터셋에서 '사람' 클래스
            # 사각형 그리기
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 클래스와 신뢰도 표시
            label = f"Person: {confidence:.2f}"
            cv2.putText(
                image,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # 결과 이미지 표시
    cv2.imshow("Detected Image", image)
    key = cv2.waitKey(0)  # 사용자 입력 대기
    if key == 27:  # ESC 키를 누르면 종료
        break

cv2.destroyAllWindows()
