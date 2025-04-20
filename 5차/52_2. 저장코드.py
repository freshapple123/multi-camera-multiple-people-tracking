# ✅ 3개 이상의 카메라를 지원하는 YOLOv8 기반 멀티카메라 실시간 Re-ID 시스템

import os
import cv2
import numpy as np
import random
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------- 설정 --------------------
COMPARE_INTERVAL = 18
SIMILARITY_THRESHOLD = 0.35
SIMILARITY_WEIGHTS = {"up": 0.3, "low": 0.3, "full": 0.2, "hair": 0.2}

# -------------------- 색상 매핑 --------------------
global_id_color_map = {}
DEFAULT_COLOR = (0, 255, 0)  # 매칭 전 기본 초록색


def get_color_for_id(gid):
    if gid not in global_id_color_map:
        global_id_color_map[gid] = (
            random.randint(60, 255),
            random.randint(60, 255),
            random.randint(60, 255),
        )
    return global_id_color_map[gid]


# -------------------- 함수 정의 --------------------
def extract_hs_histogram_from_cropped(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def get_hist_average(histories, alpha=0.2):
    if not histories:
        return None
    smoothed = histories[0]
    for hist in histories[1:]:
        smoothed = alpha * hist + (1 - alpha) * smoothed
    return smoothed


def get_valid_points(kp_array, person_idx, indices, threshold=0.5):
    return np.array(
        [
            kp_array[person_idx][i][:2]
            for i in indices
            if kp_array[person_idx][i][2] >= threshold
        ]
    )


def get_cropped_histogram(points, frame):
    if len(points) < 3:
        return None, None
    x_min, x_max = int(np.min(points[:, 0])), int(np.max(points[:, 0]))
    y_min, y_max = int(np.min(points[:, 1])), int(np.max(points[:, 1]))
    h, w, _ = frame.shape
    x_min, x_max = max(x_min, 0), min(x_max, w)
    y_min, y_max = max(y_min, 0), min(y_max, h)
    cropped = frame[y_min:y_max, x_min:x_max]
    hist = extract_hs_histogram_from_cropped(cropped)
    center = (int(np.mean(points[:, 0])), int(np.mean(points[:, 1])))
    return hist, center


def get_full_body_histogram(kp_array, person_idx, frame):
    points = get_valid_points(kp_array, person_idx, list(range(17)))
    if len(points) < 5:
        return None
    return get_cropped_histogram(points, frame)[0]


def get_hair_histogram(kp_array, person_idx, frame):
    keypoints = kp_array[person_idx]
    if np.any(keypoints[[0, 1, 2, 5, 6], 2] < 0.5):
        return None
    left_eye, right_eye = keypoints[1][:2], keypoints[2][:2]
    left_shoulder, right_shoulder = keypoints[5][:2], keypoints[6][:2]
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


def get_angle_path(n):
    angle_str = (
        f"{n}st" if n == 1 else f"{n}nd" if n == 2 else f"{n}rd" if n == 3 else f"{n}th"
    )
    return rf"D:\\REid\\data\\retail\\MMPTracking_training\\To_seperate_for_video\\{angle_str}_angle\\{angle_str}_angle.mp4"


def cross_camera_matching():
    global person_db, global_id_map, next_global_id
    cams = list(person_db.keys())
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            camA, camB = cams[i], cams[j]
            dbA, dbB = person_db[camA], person_db[camB]
            similarity_matrix = {}
            for idA, dataA in dbA.items():
                for idB, dataB in dbB.items():
                    sim_dict = {}
                    for part in ["up", "low", "full", "hair"]:
                        if dataA[f"{part}_histories"] and dataB[f"{part}_histories"]:
                            sim_dict[part] = compare_histograms(
                                get_hist_average(dataA[f"{part}_histories"]),
                                get_hist_average(dataB[f"{part}_histories"]),
                            )
                    if sim_dict:
                        values = list(sim_dict.values())
                        weights = [SIMILARITY_WEIGHTS[k] for k in sim_dict]
                        similarity_matrix[(idA, idB)] = np.average(
                            values, weights=weights
                        )

            bestB_for_A = {}
            bestA_for_B = {}
            for (idA, idB), sim in similarity_matrix.items():
                if sim > SIMILARITY_THRESHOLD:
                    if idA not in bestB_for_A or sim > similarity_matrix.get(
                        (idA, bestB_for_A[idA]), -1
                    ):
                        bestB_for_A[idA] = idB
                    if idB not in bestA_for_B or sim > similarity_matrix.get(
                        (bestA_for_B[idB], idB), -1
                    ):
                        bestA_for_B[idB] = idA

            for idA, idB in bestB_for_A.items():
                if bestA_for_B.get(idB) == idA:
                    keyA, keyB = f"{camA}_{idA}", f"{camB}_{idB}"
                    if keyA not in global_id_map and keyB not in global_id_map:
                        global_id_map[keyA] = global_id_map[keyB] = next_global_id
                        print(
                            f"🔁 Mutual 매칭됨: {keyA} ↔ {keyB} → ID: {next_global_id}"
                        )
                        next_global_id += 1
                    elif keyA in global_id_map and keyB not in global_id_map:
                        global_id_map[keyB] = global_id_map[keyA]
                    elif keyB in global_id_map and keyA not in global_id_map:
                        global_id_map[keyA] = global_id_map[keyB]


# -------------------- 초기화 --------------------
angle_ids = [1, 2, 5]
video_paths = {f"cam{i+1}": get_angle_path(angle_ids[i]) for i in range(len(angle_ids))}
caps = {cam: cv2.VideoCapture(path) for cam, path in video_paths.items()}
person_db = {cam: {} for cam in video_paths}
next_id = {cam: 0 for cam in video_paths}
global_id_map = {}
next_global_id = 0
frame_count = 0


# ▶▶▶ 저장용 VideoWriter 설정 추가
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 또는 'XVID'
fps = caps["cam1"].get(cv2.CAP_PROP_FPS) * 1.2
width = int(caps["cam1"].get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(caps["cam1"].get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path_cam1 = "output_cam1_with_tracking.mp4"  # cam1용 비디오 저장 경로
output_path_cam2 = "output_cam2_with_tracking.mp4"  # cam2용 비디오 저장 경로
output_path_cam3 = "output_cam3_with_tracking.mp4"  # cam2용 비디오 저장 경로
# 두 개의 VideoWriter 객체 생성
out_cam1 = cv2.VideoWriter(output_path_cam1, fourcc, fps, (width, height))
out_cam2 = cv2.VideoWriter(output_path_cam2, fourcc, fps, (width, height))
out_cam3 = cv2.VideoWriter(output_path_cam3, fourcc, fps, (width, height))


model = YOLO("yolov8n-pose.pt")
upper_indices = [5, 6, 11, 12]
lower_indices = [11, 12, 15, 16]

# -------------------- 메인 루프 --------------------
while True:
    frames, results = {}, {}
    for cam, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue
        frames[cam] = frame
        results[cam] = model(frame)[0].keypoints

    if not frames:
        break

    for cam in frames:
        frame = frames[cam]
        keypoints = results[cam]
        if keypoints is None or keypoints.data.shape[0] == 0:
            continue
        kp_array = keypoints.data.cpu().numpy()
        for i in range(len(kp_array)):
            up = get_valid_points(kp_array, i, upper_indices)
            low = get_valid_points(kp_array, i, lower_indices)
            up_hist, center = (
                get_cropped_histogram(up, frame) if len(up) >= 3 else (None, None)
            )
            low_hist, _ = (
                get_cropped_histogram(low, frame) if len(low) >= 3 else (None, None)
            )
            full_hist = get_full_body_histogram(kp_array, i, frame)
            hair_hist = get_hair_histogram(kp_array, i, frame)

            if all(h is None for h in [up_hist, low_hist, full_hist, hair_hist]):
                continue

            matched_id, max_sim = None, -1
            for pid, data in person_db[cam].items():
                sim_dict = {}
                for key, hist in zip(
                    ["up", "low", "full", "hair"],
                    [up_hist, low_hist, full_hist, hair_hist],
                ):
                    if hist is not None and data[f"{key}_histories"]:
                        sim_dict[key] = compare_histograms(
                            hist, get_hist_average(data[f"{key}_histories"])
                        )
                if sim_dict:
                    values = list(sim_dict.values())
                    weights = [SIMILARITY_WEIGHTS[k] for k in sim_dict]
                    sim = np.average(values, weights=weights)
                    if sim > max_sim:
                        matched_id, max_sim = pid, sim

            if matched_id is None or max_sim < SIMILARITY_THRESHOLD:
                matched_id = next_id[cam]
                next_id[cam] += 1
                person_db[cam][matched_id] = {
                    "up_histories": [],
                    "low_histories": [],
                    "full_histories": [],
                    "hair_histories": [],
                    "last_position": center,
                }

            for key, hist in zip(
                ["up", "low", "full", "hair"], [up_hist, low_hist, full_hist, hair_hist]
            ):
                if hist is not None:
                    hist_list = person_db[cam][matched_id][f"{key}_histories"]
                    hist_list.append(hist)
                    if len(hist_list) > 20:
                        hist_list.pop(0)

            gid_key = f"{cam}_{matched_id}"
            gid = global_id_map.get(gid_key, gid_key)
            if center is not None:
                color = (
                    get_color_for_id(gid) if gid_key in global_id_map else DEFAULT_COLOR
                )
                cv2.putText(
                    frame, f"ID: {gid}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

    for cam in frames:
        # 각 카메라에 대해 저장
        if cam == "cam1":
            out_cam1.write(frames[cam])  # cam1 비디오 저장
        elif cam == "cam2":
            out_cam2.write(frames[cam])  # cam2 비디오 저장
        elif cam == "cam3":
            out_cam3.write(frames[cam])  # cam2 비디오 저장

        cv2.imshow(f"Tracking - {cam}", frames[cam])

    frame_count += 1
    if frame_count % COMPARE_INTERVAL == 0:
        cross_camera_matching()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps.values():
    cap.release()


out_cam1.release()  # cam1 비디오 저장 종료
out_cam2.release()  # cam2 비디오 저장 종료
out_cam3.release()  # cam2 비디오 저장 종료
cv2.destroyAllWindows()
