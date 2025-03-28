import os
import cv2
from ultralytics import YOLO
import numpy as np

# 환경 변수 설정 (일부 라이브러리 충돌 방지)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 잘려진 이미지에서 HS 히스토그램을 추출하는 함수
def extract_hs_histogram_from_cropped(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()  # 히스토그램을 평탄화하여 반환

# 두 히스토그램 간의 유사도를 비교하는 함수
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# 장기 히스토그램(최대 10개)과 단기 히스토그램(최대 3개) 간의 유사도를 비교하는 함수
def compare_histograms_weighted(short_hist1, short_hist2, long_hist1, long_hist2, short_weight=0.7, long_weight=0.3):
    short_similarity = compare_histograms(short_hist1, short_hist2)
    long_similarity = compare_histograms(long_hist1, long_hist2)
    weighted_similarity = short_weight * short_similarity + long_weight * long_similarity
    return weighted_similarity

# 최근 N개의 히스토그램을 기반으로 지수 이동 평균 적용
def get_hist_average(histories, alpha=0.2):
    if len(histories) == 0:
        return None
    smoothed_hist = histories[0]
    for hist in histories[1:]:
        smoothed_hist = alpha * hist + (1 - alpha) * smoothed_hist
    return smoothed_hist

# 키포인트 정보를 바탕으로 잘려진 이미지 및 히스토그램을 추출하는 함수
def get_cropped_and_histogram(kp_array, frame, person_idx, desired_indices):
    valid_points = []
    for idx in desired_indices:
        x, y, conf = kp_array[person_idx][idx]
        if conf >= 0.5:
            valid_points.append([x, y])

    valid_points = np.array(valid_points)
    if len(valid_points) >= 3:
        x_min, x_max = int(np.min(valid_points[:, 0])), int(np.max(valid_points[:, 0]))
        y_min, y_max = int(np.min(valid_points[:, 1])), int(np.max(valid_points[:, 1]))
        h, w, _ = frame.shape
        x_min, x_max = max(x_min, 0), min(x_max, w)
        y_min, y_max = max(y_min, 0), min(y_max, h)

        cropped_img = frame[y_min:y_max, x_min:x_max]
        hist = extract_hs_histogram_from_cropped(cropped_img)
        center_x, center_y = int(np.mean(valid_points[:, 0])), int(np.mean(valid_points[:, 1]))

        return cropped_img, hist, (center_x, center_y)
    return None, None, None

# 비디오 파일 경로 설정
video_path = r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\2nd_angle\2nd_angle.mp4"

cap = cv2.VideoCapture(video_path)

# YOLO 모델 로드
model = YOLO("yolov8n-pose.pt")
# 추적할 키포인트 인덱스 설정 (어깨, 엉덩이 부분)
desired_indices = [5, 6, 11, 12]

# 인물 데이터베이스 및 ID 초기화
person_db = {}
next_id = 0

print("▶ 영상 분석 시작... 'Q' 키로 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    keypoints = results[0].keypoints

    if keypoints is None or keypoints.data.shape[0] == 0:
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    kp_array = keypoints.data.cpu().numpy()
    current_ids = set()

    for person_idx in range(len(kp_array)):
        cropped_img, hist, center_pos = get_cropped_and_histogram(
            kp_array, frame, person_idx, desired_indices
        )
        if hist is None:
            continue

        matched_id = None
        max_similarity = -1

        # 기존 ID 중에서 가장 유사한 ID 찾기
        for pid, data in person_db.items():
            # 장기 히스토그램과 단기 히스토그램 비교
            long_hist = get_hist_average(data["histories"], alpha=0.2)
            short_hist = data["histories"][-1] if len(data["histories"]) > 0 else None

            if long_hist is not None and short_hist is not None:
                similarity = compare_histograms_weighted(hist, short_hist, long_hist, hist)
                if similarity > max_similarity:
                    matched_id = pid
                    max_similarity = similarity

        # 새로운 사람인 경우 ID 부여
        if matched_id is None or max_similarity < 0.3:
            matched_id = next_id
            next_id += 1
            person_db[matched_id] = {"histories": [], "last_position": center_pos}

        # 히스토그램 저장
        person_db[matched_id]["histories"].append(hist)

        # 오래된 히스토그램 유지 (최대 10개)(각 사람마다)
        if len(person_db[matched_id]["histories"]) > 10:
            person_db[matched_id]["histories"].pop(0)

        person_db[matched_id]["last_position"] = center_pos
        current_ids.add(matched_id)

        # 화면에 ID 표시
        if center_pos is not None:
            cv2.putText(
                frame,
                f"ID: {matched_id}",
                center_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
