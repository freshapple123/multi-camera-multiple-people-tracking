"""
내 생각에는 다른 각도에서의 사람을 찾을때 말야,
다른 각도의 카메라서 계속해서 서로 유사도를 검사하는게 아니라
첨에 제일 비슷한 사람이 있으면 id번호만 넘겨주고
나중에 트레킹 및 re-id 하는건 각자 각도의 히스토유사도검사로
원래 하듯이 계속 트레킹 하는거지
그러면 비슷한 사람이 있어도 계속 separate시킬 수 있을거고
re-id부여도 좋을듯

"""

import os
import cv2
from ultralytics import YOLO
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------- 설정 --------------------
COMPARE_INTERVAL = 30  # 약 3초마다 cross-camera ID 비교 수행 (30fps 기준)
SIMILARITY_THRESHOLD = 0.4  # 유사도 임계값 (이 값 이상이면 동일 인물로 판단)
SIMILARITY_WEIGHTS = {
    "up": 0.3,  # 상체 영역 히스토그램 가중치
    "low": 0.3,  # 하체 영역 히스토그램 가중치
    "full": 0.2,  # 전체 몸통 히스토그램 가중치
    "hair": 0.2,  # 머리 영역 히스토그램 가중치
}


# -------------------- 함수 정의 --------------------


def extract_hs_histogram_from_cropped(cropped_img):
    # 이미지의 색상 정보를 HSV로 변환한 후, H-S 채널 기준 2D 히스토그램 추출
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None, [50, 60], [0, 180, 0, 256]
    )  # H: 50 bins, S: 60 bins
    cv2.normalize(hist, hist)  # 정규화하여 조명이나 크기에 덜 민감하게 만듦
    return hist.flatten()  # 1차원 벡터로 변환해 반환


def compare_histograms(hist1, hist2):
    # 두 히스토그램 간의 유사도를 상관계수(CORREL) 방식으로 비교
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def get_hist_average(histories, alpha=0.2):
    # 과거 히스토리들을 지수평균 방식으로 스무딩 (최근 히스토리에 가중치 부여)
    if len(histories) == 0:
        return None
    smoothed_hist = histories[0]
    for hist in histories[1:]:
        smoothed_hist = alpha * hist + (1 - alpha) * smoothed_hist  # 지수 가중 평균
    return smoothed_hist


def get_valid_points(kp_array, person_idx, indices, threshold=0.5):
    # 신뢰도(threshold) 이상인 keypoint들만 추출해서 반환
    points = []
    for idx in indices:
        x, y, conf = kp_array[person_idx][idx]
        if conf >= threshold:
            points.append([x, y])
    return np.array(points)


def get_cropped_histogram(points, frame):
    # 주어진 keypoint들 좌표 기반으로 bounding box 만들고, 해당 영역의 HS 히스토그램 추출
    if len(points) < 3:
        return None, None  # 최소 3점 이상 있어야 의미 있는 영역이라 판단
    x_min, x_max = int(np.min(points[:, 0])), int(np.max(points[:, 0]))
    y_min, y_max = int(np.min(points[:, 1])), int(np.max(points[:, 1]))
    h, w, _ = frame.shape
    x_min, x_max = max(x_min, 0), min(x_max, w)
    y_min, y_max = max(y_min, 0), min(y_max, h)
    cropped_img = frame[y_min:y_max, x_min:x_max]
    hist = extract_hs_histogram_from_cropped(cropped_img)  # 잘라낸 영역의 HS 히스토그램
    center_x, center_y = int(np.mean(points[:, 0])), int(
        np.mean(points[:, 1])
    )  # 중심 좌표도 함께 반환
    return hist, (center_x, center_y)


def get_full_body_histogram(kp_array, person_idx, frame):
    # 17개의 모든 keypoint를 이용해 전체 몸 영역 bounding box 계산 후 히스토그램 추출
    full_points = get_valid_points(kp_array, person_idx, list(range(17)))
    if len(full_points) < 5:
        return None  # 전체 몸 인식이 제대로 안 된 경우는 제외
    x_min, x_max = int(np.min(full_points[:, 0])), int(np.max(full_points[:, 0]))
    y_min, y_max = int(np.min(full_points[:, 1])), int(np.max(full_points[:, 1]))
    h, w, _ = frame.shape
    x_min, x_max = max(x_min, 0), min(x_max, w)
    y_min, y_max = max(y_min, 0), min(y_max, h)
    cropped_img = frame[y_min:y_max, x_min:x_max]
    return extract_hs_histogram_from_cropped(cropped_img)


def get_hair_histogram(kp_array, person_idx, frame):
    # 얼굴과 어깨 keypoint를 활용하여 머리 부분 영역을 잘라내고 HS 히스토그램을 추출
    keypoints = kp_array[person_idx]
    # 왼눈, 오른눈, 왼어깨, 오른어깨, 코 등의 keypoint 신뢰도가 낮으면 제외
    if np.any(keypoints[[0, 1, 2, 5, 6], 2] < 0.5):
        return None

    # 눈과 어깨 좌표 추출
    left_eye, right_eye = keypoints[1][:2], keypoints[2][:2]
    left_shoulder, right_shoulder = keypoints[5][:2], keypoints[6][:2]

    # 머리 영역의 bounding box 좌표 계산 (눈 위쪽부터 어깨 위까지)
    x_min = int(min(left_eye[0], right_eye[0]) - 30)
    x_max = int(max(left_eye[0], right_eye[0]) + 30)
    y_top = int(min(left_eye[1], right_eye[1]) - 30)
    y_bottom = int(min(left_shoulder[1], right_shoulder[1]) - 10)

    h, w, _ = frame.shape
    x_min, x_max = max(x_min, 0), min(x_max, w)
    y_top, y_bottom = max(y_top, 0), min(y_bottom, h)

    # 유효하지 않은 영역이면 None 반환
    if y_bottom <= y_top or x_max <= x_min:
        return None

    # 머리 영역 잘라내기
    cropped = frame[y_top:y_bottom, x_min:x_max]
    return extract_hs_histogram_from_cropped(cropped)


def get_angle_path(n):
    # 입력된 숫자(n)에 따라 n번째 각도의 영상 경로를 반환
    angle_str = (
        f"{n}st" if n == 1 else f"{n}nd" if n == 2 else f"{n}rd" if n == 3 else f"{n}th"
    )
    return rf"D:\\REid\\data\\retail\\MMPTracking_training\\To_seperate_for_video\\{angle_str}_angle\\{angle_str}_angle.mp4"


def compute_similarity(hist1, hist2):
    # 두 히스토그램이 유효한 경우 유사도 계산, 아니라면 None 반환
    if hist1 is None or hist2 is None:
        return None
    return compare_histograms(hist1, hist2)


def cross_camera_matching():
    global person_db, global_id_map, next_global_id  # 전역 변수 사용

    # 각 카메라에서 추출된 사람 데이터
    cam1_db = person_db["cam1"]
    cam2_db = person_db["cam2"]

    similarity_matrix = {}  # (cam1_id, cam2_id) → 유사도 저장

    # -------------------- 모든 ID 쌍에 대해 유사도 계산 --------------------
    for id1, data1 in cam1_db.items():
        for id2, data2 in cam2_db.items():
            sim_dict = {}  # 상체, 하체, 전체, 머리 별 유사도 저장

            # 각 부위별로 히스토리 존재할 경우 유사도 계산
            if data1["up_histories"] and data2["up_histories"]:
                sim_dict["up"] = compare_histograms(
                    get_hist_average(data1["up_histories"]),
                    get_hist_average(data2["up_histories"]),
                )
            if data1["low_histories"] and data2["low_histories"]:
                sim_dict["low"] = compare_histograms(
                    get_hist_average(data1["low_histories"]),
                    get_hist_average(data2["low_histories"]),
                )
            if data1["full_histories"] and data2["full_histories"]:
                sim_dict["full"] = compare_histograms(
                    get_hist_average(data1["full_histories"]),
                    get_hist_average(data2["full_histories"]),
                )
            if data1["hair_histories"] and data2["hair_histories"]:
                sim_dict["hair"] = compare_histograms(
                    get_hist_average(data1["hair_histories"]),
                    get_hist_average(data2["hair_histories"]),
                )

            # 하나라도 유사도가 있으면 가중 평균 계산
            if sim_dict:
                sim_values = list(sim_dict.values())  # 유사도 값 리스트
                weight_values = [
                    SIMILARITY_WEIGHTS[k] for k in sim_dict.keys()
                ]  # 해당 부위 가중치
                similarity = np.average(sim_values, weights=weight_values)
                similarity_matrix[(id1, id2)] = similarity  # 유사도 저장

    # -------------------- cam1 → cam2: 가장 유사한 ID --------------------
    best_cam2_for_cam1 = {}
    for (id1, id2), sim in similarity_matrix.items():
        if sim > SIMILARITY_THRESHOLD:  # 임계값 이상인 경우만 고려
            # 현재까지의 최고 유사도보다 높으면 업데이트
            if id1 not in best_cam2_for_cam1 or sim > similarity_matrix.get(
                (id1, best_cam2_for_cam1[id1]), -1
            ):
                best_cam2_for_cam1[id1] = id2

    # -------------------- cam2 → cam1: 가장 유사한 ID --------------------
    best_cam1_for_cam2 = {}
    for (id1, id2), sim in similarity_matrix.items():
        if sim > SIMILARITY_THRESHOLD:
            if id2 not in best_cam1_for_cam2 or sim > similarity_matrix.get(
                (best_cam1_for_cam2[id2], id2), -1
            ):
                best_cam1_for_cam2[id2] = id1

    # -------------------- Mutual Best Matching --------------------
    for id1, id2 in best_cam2_for_cam1.items():
        if (
            best_cam1_for_cam2.get(id2) == id1
        ):  # 서로가 서로를 가장 유사하다고 판단한 경우
            key1 = f"cam1_{id1}"
            key2 = f"cam2_{id2}"

            # 두 ID 모두 처음 매칭되는 경우: 새 global ID 부여
            if key1 not in global_id_map and key2 not in global_id_map:
                global_id_map[key1] = next_global_id
                global_id_map[key2] = next_global_id
                print(
                    f"🔁 Mutual 매칭됨: cam1_{id1} ↔ cam2_{id2} → ID: {next_global_id}"
                )
                next_global_id += 1

            # 한쪽만 global_id가 있으면 다른 쪽에 복사
            elif key1 in global_id_map and key2 not in global_id_map:
                global_id_map[key2] = global_id_map[key1]
            elif key2 in global_id_map and key1 not in global_id_map:
                global_id_map[key1] = global_id_map[key2]


# -------------------- 메인 실행 --------------------

a, b = 1, 5  # 사용할 카메라 각도 번호 (예: 1번 각도와 5번 각도 비교)
video_paths = {
    "cam1": get_angle_path(a),  # cam1 영상 경로
    "cam2": get_angle_path(b),  # cam2 영상 경로
}

# 각 비디오 파일을 OpenCV VideoCapture 객체로 로드
caps = {
    "cam1": cv2.VideoCapture(video_paths["cam1"]),
    "cam2": cv2.VideoCapture(video_paths["cam2"]),
}

# YOLOv8 포즈 모델 로드 (경량 모델 사용)
model = YOLO("yolov8n-pose.pt")

# 신체 부위 인덱스 (keypoint index 기준)
upper_indices = [5, 6, 11, 12]  # 양쪽 어깨와 엉덩이 (상체)
lower_indices = [11, 12, 15, 16]  # 양쪽 엉덩이와 발목 (하체)

# 각 카메라 별 사람 정보 저장 딕셔너리 (히스토리 포함)
person_db = {"cam1": {}, "cam2": {}}

# 각 카메라 별 다음에 부여할 로컬 ID
next_id = {"cam1": 0, "cam2": 0}

# cross-camera 통합 ID 매핑: "cam1_0" → 1, "cam2_3" → 1 같은 형식
global_id_map = {}

# 다음에 부여할 global ID 번호 (mutual match 시 할당됨)
next_global_id = 0

# 전체 프레임 카운트 (interval마다 cross-camera 비교용)
frame_count = 0

while True:
    frames = {}
    results = {}

    # -------------------- 각 카메라에서 프레임 읽고, YOLO Pose 추론 --------------------
    for cam_name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue
        frames[cam_name] = frame
        results[cam_name] = model(frame)[0].keypoints  # 포즈 keypoints 추출

    if not frames:
        break  # 모든 카메라 프레임이 끝났을 경우 종료

    # -------------------- 카메라별 프레임 처리 --------------------
    for cam_name in frames:
        frame = frames[cam_name]
        keypoints = results[cam_name]
        if keypoints is None or keypoints.data.shape[0] == 0:
            continue  # 사람 인식 실패 시 건너뜀

        kp_array = keypoints.data.cpu().numpy()  # keypoints를 numpy 배열로 변환

        # -------------------- 프레임 내 모든 사람에 대해 처리 --------------------
        for person_idx in range(len(kp_array)):
            # 상체/하체 keypoint 좌표 추출
            up_points = get_valid_points(kp_array, person_idx, upper_indices)
            low_points = get_valid_points(kp_array, person_idx, lower_indices)

            # 상체/하체 histogram과 중심좌표 계산
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

            # 전체/머리 histogram 추출
            full_hist = get_full_body_histogram(kp_array, person_idx, frame)
            hair_hist = get_hair_histogram(kp_array, person_idx, frame)

            # 4개 다 None이면 무시
            if all(h is None for h in [up_hist, low_hist, full_hist, hair_hist]):
                continue

            # -------------------- 기존 person과 유사도 비교 (re-ID) --------------------
            matched_id = None
            max_similarity = -1
            for pid, data in person_db[cam_name].items():
                sim_dict = {}

                if up_hist is not None and data["up_histories"]:
                    sim_dict["up"] = compare_histograms(
                        up_hist, get_hist_average(data["up_histories"])
                    )
                if low_hist is not None and data["low_histories"]:
                    sim_dict["low"] = compare_histograms(
                        low_hist, get_hist_average(data["low_histories"])
                    )
                if full_hist is not None and data["full_histories"]:
                    sim_dict["full"] = compare_histograms(
                        full_hist, get_hist_average(data["full_histories"])
                    )
                if hair_hist is not None and data["hair_histories"]:
                    sim_dict["hair"] = compare_histograms(
                        hair_hist, get_hist_average(data["hair_histories"])
                    )

                # 유사도 가중 평균 계산
                if sim_dict:
                    sim_values = list(sim_dict.values())
                    weight_values = [SIMILARITY_WEIGHTS[k] for k in sim_dict.keys()]
                    similarity = np.average(sim_values, weights=weight_values)

                    if similarity > max_similarity:
                        matched_id = pid
                        max_similarity = similarity

            # -------------------- 새로운 사람으로 간주 --------------------
            if matched_id is None or max_similarity < SIMILARITY_THRESHOLD:
                matched_id = next_id[cam_name]
                next_id[cam_name] += 1
                person_db[cam_name][matched_id] = {
                    "up_histories": [],
                    "low_histories": [],
                    "full_histories": [],
                    "hair_histories": [],
                    "last_position": center_pos,
                }

            # -------------------- 히스토리 업데이트 --------------------
            for key, hist in zip(
                ["up", "low", "full", "hair"], [up_hist, low_hist, full_hist, hair_hist]
            ):
                if hist is not None:
                    hist_list = person_db[cam_name][matched_id][f"{key}_histories"]
                    hist_list.append(hist)
                    if len(hist_list) > 20:
                        hist_list.pop(0)  # 최대 20개까지만 저장 (메모리 제한)

            # -------------------- ID 시각화 --------------------
            global_key = f"{cam_name}_{matched_id}"
            global_id = global_id_map.get(
                global_key, global_key
            )  # global ID가 있으면 출력, 없으면 로컬 ID
            if center_pos is not None:
                cv2.putText(
                    frame,
                    f"ID: {global_id}",
                    center_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

    # -------------------- 결과 프레임 디스플레이 --------------------
    for cam_name in frames:
        cv2.imshow(f"Tracking - {cam_name}", frames[cam_name])

    # -------------------- 주기적으로 cross-camera 매칭 --------------------
    frame_count += 1
    if frame_count % COMPARE_INTERVAL == 0:
        cross_camera_matching()

    # -------------------- 종료 조건 --------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------- 종료 처리 --------------------
for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
