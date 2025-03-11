from ultralytics import YOLO
import cv2

# YOLO 모델 불러오기 (사전 학습된 가중치 사용)
model = YOLO("yolov8n.pt")  # 'yolov8n.pt'는 경량 모델, 필요에 따라 다른 모델 선택 가능

# 분석할 이미지 경로
image_path = "D:/RE-id/data/retail/MMPTracking_training/retail_0/rgb_00000_1.jpg"

# 이미지 불러오기
image = cv2.imread(image_path)

# YOLO 모델로 이미지 예측
results = model(image)

# 예측된 객체 정보 가져오기
detections = results[0].boxes.data.cpu().numpy()  # YOLOv8에서 박스 정보 가져오기
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
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과 저장 (선택 사항)
cv2.imwrite("output_image.jpg", image)
