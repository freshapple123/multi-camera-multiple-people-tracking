import os
import cv2
from ultralytics import YOLO
import numpy as np

# OpenMP 충돌 방지 설정 (Windows에서 일부 OpenCV 버전과 PyTorch 조합 시 충돌 방지)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# -------------------- 함수 정의 --------------------


# HS 히스토그램 추출 함수 (Hue, Saturation 채널 기반)
def extract_hs_histogram_from_cropped(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)  # BGR → HSV 변환
    hist = cv2.calcHist(
        [hsv], [0, 1], None, [50, 60], [0, 180, 0, 256]
    )  # HS 2D 히스토그램
    cv2.normalize(hist, hist)  # 정규화
    return hist.flatten()  # 1차원 벡터로 반환


# 히스토그램 비교 함수 (상관계수 방식)
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# EMA 방식으로 히스토리 평균 내기 (지수평활법으로 최신 값 가중치 높임)
def get_hist_average(histories, alpha=0.2):
    if len(histories) == 0:
        return None
    smoothed_hist = histories[0]
    for hist in histories[1:]:
        smoothed_hist = alpha * hist + (1 - alpha) * smoothed_hist
    return smoothed_hist


# 키포인트에서 특정 인덱스 좌표만 유효 confidence 기준으로 추출
def get_valid_points(kp_array, person_idx, indices, threshold=0.5):
    points = []
    for idx in indices:
        x, y, conf = kp_array[person_idx][idx]
        if conf >= threshold:
            points.append([x, y])
    return np.array(points)


# 지정된 포인트 영역에서 이미지 crop 후 히스토그램과 중심 좌표 추출
def get_cropped_histogram(points, frame):
    if len(points) < 3:
        return None, None
    x_min, x_max = int(np.min(points[:, 0])), int(np.max(points[:, 0]))
    y_min, y_max = int(np.min(points[:, 1])), int(np.max(points[:, 1]))
    h, w, _ = frame.shape
    x_min, x_max = max(x_min, 0), min(x_max, w)
    y_min, y_max = max(y_min, 0), min(y_max, h)
    cropped_img = frame[y_min:y_max, x_min:x_max]
    hist = extract_hs_histogram_from_cropped(cropped_img)
    center_x, center_y = int(np.mean(points[:, 0])), int(np.mean(points[:, 1]))
    return hist, (center_x, center_y)


# -------------------- 영상 및 모델 로딩 --------------------

# 분석할 영상 설정 (1~5 선택 가능)
a = 3
if a == 1:
    video_path = r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\1st_angle\1st_angle.mp4"
elif a == 2:
    video_path = r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\2nd_angle\2nd_angle.mp4"
elif a == 3:
    video_path = r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\3rd_angle\3rd_angle.mp4"
elif a == 4:
    video_path = r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\4th_angle\4th_angle.mp4"
elif a == 5:
    video_path = r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\5th_angle\5th_angle.mp4"

cap = cv2.VideoCapture(video_path)
model = YOLO("yolov8n-pose.pt")  # 경량화된 YOLOv8 포즈 모델

# 포즈 키포인트 인덱스 정의
upper_indices = [5, 6, 11, 12]  # 어깨와 골반 (상체)
lower_indices = [11, 12, 15, 16]  # 골반과 발 (하체)

# ID 매핑을 위한 person 데이터베이스 초기화
person_db = {}
next_id = 0

print("▶ 영상 분석 시작... 'Q' 키로 종료")

# -------------------- 프레임별 분석 시작 --------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    keypoints = results[0].keypoints

    # 사람이 감지되지 않은 경우
    if keypoints is None or keypoints.data.shape[0] == 0:
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    kp_array = keypoints.data.cpu().numpy()
    current_ids = set()

    for person_idx in range(len(kp_array)):
        # 상체, 하체 포인트 추출
        up_points = get_valid_points(kp_array, person_idx, upper_indices)
        low_points = get_valid_points(kp_array, person_idx, lower_indices)

        # 상체/하체에서 크롭된 히스토그램과 중심좌표 추출
        up_hist, center_pos = (
            get_cropped_histogram(up_points, frame)
            if len(up_points) >= 3
            else (None, None)
        )
        low_hist, _ = (
            get_cropped_histogram(low_points, frame)
            if len(low_points) >= 3
            else (None, None)
        )

        # 상하체 히스토그램이 모두 없는 경우는 무시
        if up_hist is None and low_hist is None:
            continue

        matched_id = None
        max_similarity = -1

        for pid, data in person_db.items():
            sim_list = []

            # 상체 히스토그램 유사도 비교
            if up_hist is not None and len(data["up_histories"]) > 0:
                avg_up = get_hist_average(data["up_histories"])
                sim_list.append(compare_histograms(up_hist, avg_up))

            # 하체 히스토그램 유사도 비교
            if low_hist is not None and len(data["low_histories"]) > 0:
                avg_low = get_hist_average(data["low_histories"])
                sim_list.append(compare_histograms(low_hist, avg_low))

            # 가중 평균 유사도로 결정
            if len(sim_list) > 0:
                if len(sim_list) == 2:
                    similarity = 0.5 * sim_list[0] + 0.5 * sim_list[1]
                else:
                    similarity = sim_list[0]
                if similarity > max_similarity:
                    matched_id = pid
                    max_similarity = similarity

        # 새 사람으로 판단되는 경우
        if (
            matched_id is None or max_similarity < 0.3
        ):  # 유사도가 낮으면 새로운 사람으로 간주
            matched_id = next_id
            next_id += 1
            person_db[matched_id] = {
                "up_histories": [],
                "low_histories": [],
                "last_position": center_pos,
                "frames_since_seen": 0,
            }

        # 상체 히스토리 저장
        if up_hist is not None:
            person_db[matched_id]["up_histories"].append(up_hist)
            if len(person_db[matched_id]["up_histories"]) > 10:
                person_db[matched_id]["up_histories"].pop(0)

        # 하체 히스토리 저장
        if low_hist is not None:
            person_db[matched_id]["low_histories"].append(low_hist)
            if len(person_db[matched_id]["low_histories"]) > 10:
                person_db[matched_id]["low_histories"].pop(0)

        # 마지막 위치 갱신
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

    # 프레임 출력
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
