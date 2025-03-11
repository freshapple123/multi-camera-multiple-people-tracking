import cv2
import logging
from ultralytics import YOLO

# 로그 출력 레벨을 설정하여 YOLO의 로그를 숨기기
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# YOLOv8 모델 로드 (사람 감지에 적합한 COCO 사전 학습 모델)
model = YOLO("yolo11n.pt")  # YOLOv8n은 속도가 빠름

# 비디오 파일 경로
video_path = r"D:/REid/data/retail/MMPTracking_training/To_seperate_for_video/1st_angle/1st_angle.mp4"
cap = cv2.VideoCapture(video_path)

# 영상이 정상적으로 열리는지 확인
if not cap.isOpened():
    print("비디오를 열 수 없습니다.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 영상 끝

    # YOLO를 사용하여 객체 감지
    results = model(frame)

    # 감지된 객체를 영상에 표시
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())  # 클래스 인덱스

            # 클래스가 'person'(사람)일 경우에만 표시 (COCO 데이터셋에서 person은 class 0)
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 결과 영상 출력
    cv2.imshow("YOLOv8 Person Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
