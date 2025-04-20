"""
가중치도 다르게 했다 제목 너무 길어서...
"""

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


def get_angle_path(n):
    angle_str = (
        f"{n}st" if n == 1 else f"{n}nd" if n == 2 else f"{n}rd" if n == 3 else f"{n}th"
    )
    return rf"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\{angle_str}_angle\{angle_str}_angle.mp4"


def get_full_body_histogram(kp_array, person_idx, frame):
    full_points = get_valid_points(kp_array, person_idx, list(range(17)), threshold=0.5)
    if len(full_points) < 5:
        return None
    x_min, x_max = int(np.min(full_points[:, 0])), int(np.max(full_points[:, 0]))
    y_min, y_max = int(np.min(full_points[:, 1])), int(np.max(full_points[:, 1]))
    h, w, _ = frame.shape
    x_min, x_max = max(x_min, 0), min(x_max, w)
    y_min, y_max = max(y_min, 0), min(y_max, h)
    cropped_img = frame[y_min:y_max, x_min:x_max]
    return extract_hs_histogram_from_cropped(cropped_img)


def get_hair_histogram(kp_array, person_idx, frame):
    keypoints = kp_array[person_idx]
    if np.any(keypoints[[0, 1, 2, 5, 6], 2] < 0.5):
        return None
    left_eye = keypoints[1][:2]
    right_eye = keypoints[2][:2]
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]

    x_min = int(min(left_eye[0], right_eye[0]) - 30)
    x_max = int(max(left_eye[0], right_eye[0]) + 30)
    y_top = int(min(left_eye[1], right_eye[1]) - 30)
    y_bottom = int(min(left_shoulder[1], right_shoulder[1]) - 10)

    h, w, _ = frame.shape
    x_min, x_max = max(x_min, 0), min(x_max, w)
    y_top, y_bottom = max(y_top, 0), min(y_bottom, h)

    if y_bottom <= y_top or x_max <= x_min:
        return None

    cropped = frame[y_top:y_bottom, x_min:x_max]
    return extract_hs_histogram_from_cropped(cropped)


# -------------------- 영상 및 모델 로딩 --------------------
a = 1
b = 2
video_paths = {"cam1": get_angle_path(a), "cam2": get_angle_path(b)}


caps = {
    "cam1": cv2.VideoCapture(video_paths["cam1"]),
    "cam2": cv2.VideoCapture(video_paths["cam2"]),
}

model = YOLO("yolov8n-pose.pt")  # 경량화된 YOLOv8 포즈 모델

# 포즈 키포인트 인덱스 정의
upper_indices = [5, 6, 11, 12]  # 어깨와 골반 (상체)
lower_indices = [11, 12, 15, 16]  # 골반과 발 (하체)

# ID 매핑을 위한 person 데이터베이스 및 ID 카운터 초기화 (영상별로 따로)
person_db = {"cam1": {}, "cam2": {}}
next_id = {"cam1": 0, "cam2": 0}


print("▶ 영상 분석 시작... 'Q' 키로 종료")

# -------------------- 프레임별 분석 시작 --------------------

while True:
    frames = {}
    results = {}

    for cam_name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue
        frames[cam_name] = frame
        results[cam_name] = model(frame)[0].keypoints

    if not frames:
        break

    for cam_name in frames.keys():
        frame = frames[cam_name]
        keypoints = results[cam_name]

        if keypoints is None or keypoints.data.shape[0] == 0:
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
            full_hist = get_full_body_histogram(kp_array, person_idx, frame)
            hair_hist = get_hair_histogram(kp_array, person_idx, frame)

            # 상하체 히스토그램이 모두 없는 경우는 무시
            if (
                up_hist is None
                and low_hist is None
                and full_hist is None
                and hair_hist is None
            ):
                continue

            matched_id = None
            max_similarity = -1

            # 영상별로 별도의 person_db 사용
            for pid, data in person_db[cam_name].items():
                sim_dict = {}
                weights = {"up": 0.3, "low": 0.3, "full": 0.2, "hair": 0.2}

                # 유사도 계산
                if up_hist is not None and len(data["up_histories"]) > 0:
                    avg_up = get_hist_average(data["up_histories"])
                    sim_dict["up"] = compare_histograms(up_hist, avg_up)

                if low_hist is not None and len(data["low_histories"]) > 0:
                    avg_low = get_hist_average(data["low_histories"])
                    sim_dict["low"] = compare_histograms(low_hist, avg_low)

                if full_hist is not None and len(data["full_histories"]) > 0:
                    avg_full = get_hist_average(data["full_histories"])
                    sim_dict["full"] = compare_histograms(full_hist, avg_full)

                if hair_hist is not None and len(data["hair_histories"]) > 0:
                    avg_hair = get_hist_average(data["hair_histories"])
                    sim_dict["hair"] = compare_histograms(hair_hist, avg_hair)

                # 가중 평균 계산
                if len(sim_dict) > 0:
                    sim_values = list(sim_dict.values())
                    weight_values = [weights[k] for k in sim_dict.keys()]
                    similarity = np.average(sim_values, weights=weight_values)

                    if similarity > max_similarity:
                        matched_id = pid
                        max_similarity = similarity

            # 새 사람으로 판단되는 경우
            if matched_id is None or max_similarity < 0.3:
                matched_id = next_id[cam_name]
                next_id[cam_name] += 1
                person_db[cam_name][matched_id] = {
                    "up_histories": [],
                    "low_histories": [],
                    "full_histories": [],
                    "hair_histories": [],
                    "last_position": center_pos,
                    "frames_since_seen": 0,
                }

            if up_hist is not None:
                person_db[cam_name][matched_id]["up_histories"].append(up_hist)
                if len(person_db[cam_name][matched_id]["up_histories"]) > 10:
                    person_db[cam_name][matched_id]["up_histories"].pop(0)

            if low_hist is not None:
                person_db[cam_name][matched_id]["low_histories"].append(low_hist)
                if len(person_db[cam_name][matched_id]["low_histories"]) > 10:
                    person_db[cam_name][matched_id]["low_histories"].pop(0)

            if full_hist is not None:
                person_db[cam_name][matched_id]["full_histories"].append(full_hist)
                if len(person_db[cam_name][matched_id]["full_histories"]) > 20:
                    person_db[cam_name][matched_id]["full_histories"].pop(0)

            if hair_hist is not None:
                person_db[cam_name][matched_id]["hair_histories"].append(hair_hist)
                if len(person_db[cam_name][matched_id]["hair_histories"]) > 20:
                    person_db[cam_name][matched_id]["hair_histories"].pop(0)

            person_db[cam_name][matched_id]["last_position"] = center_pos
            current_ids.add(matched_id)

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

    # -------- 화면 출력 --------
    for cam_name in frames:
        cv2.imshow(f"Tracking - {cam_name}", frames[cam_name])

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------- 종료 처리 --------
for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
