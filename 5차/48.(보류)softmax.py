"""
마지막 합칠 때, 소프트맥스 처럼 유사도를 일정한 비율로 해보자
"""

import os
import cv2
from ultralytics import YOLO
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------- 유틸 함수들 -----------


def extract_hs_histogram_from_cropped(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def get_hist_average(histories, alpha=0.2):
    if len(histories) == 0:
        return None
    smoothed_hist = histories[0]
    for hist in histories[1:]:
        smoothed_hist = alpha * hist + (1 - alpha) * smoothed_hist
    return smoothed_hist


def get_valid_points(kp_array, person_idx, indices, threshold=0.5):
    points = []
    for idx in indices:
        x, y, conf = kp_array[person_idx][idx]
        if conf >= threshold:
            points.append([x, y])
    return np.array(points)


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


def softmax(x, temperature=1.0):
    x = np.array(x)
    exp_x = np.exp(x / temperature)
    return exp_x / np.sum(exp_x)


# ----------- 설정 -----------

a = 1
b = 5
video_paths = {"cam1": get_angle_path(a), "cam2": get_angle_path(b)}

caps = {
    "cam1": cv2.VideoCapture(video_paths["cam1"]),
    "cam2": cv2.VideoCapture(video_paths["cam2"]),
}

model = YOLO("yolov8n-pose.pt")

upper_indices = [5, 6, 11, 12]
lower_indices = [11, 12, 15, 16]

person_db = {"cam1": {}, "cam2": {}}
next_id = {"cam1": 0, "cam2": 0}

print("▶ 영상 분석 시작... 'Q' 키로 종료")

# ----------- 분석 루프 -----------

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
            up_points = get_valid_points(kp_array, person_idx, upper_indices)
            low_points = get_valid_points(kp_array, person_idx, lower_indices)

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

            if up_hist is None and low_hist is None:
                continue

            matched_id = None
            max_similarity = -1

            for pid, data in person_db[cam_name].items():
                sim_list = []
                if up_hist is not None and len(data["up_histories"]) > 0:
                    avg_up = get_hist_average(data["up_histories"])
                    sim_list.append(compare_histograms(up_hist, avg_up))
                if low_hist is not None and len(data["low_histories"]) > 0:
                    avg_low = get_hist_average(data["low_histories"])
                    sim_list.append(compare_histograms(low_hist, avg_low))

                if len(sim_list) > 0:
                    similarity = np.mean(sim_list)
                    if similarity > max_similarity:
                        matched_id = pid
                        max_similarity = similarity

            if matched_id is None or max_similarity < 0.3:
                matched_id = next_id[cam_name]
                next_id[cam_name] += 1
                person_db[cam_name][matched_id] = {
                    "up_histories": [],
                    "low_histories": [],
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

    # ----------- 양방향 소프트맥스 매칭 -----------

    best_matches_1to2 = {}
    for id1, data1 in person_db["cam1"].items():
        up1 = get_hist_average(data1["up_histories"])
        low1 = get_hist_average(data1["low_histories"])
        if up1 is None and low1 is None:
            continue
        similarities = []
        id2_list = []
        for id2, data2 in person_db["cam2"].items():
            up2 = get_hist_average(data2["up_histories"])
            low2 = get_hist_average(data2["low_histories"])
            if up2 is None and low2 is None:
                continue
            sim_list = []
            if up1 is not None and up2 is not None:
                sim_list.append(compare_histograms(up1, up2))
            if low1 is not None and low2 is not None:
                sim_list.append(compare_histograms(low1, low2))
            if len(sim_list) == 0:
                continue
            similarity = np.mean(sim_list)
            similarities.append(similarity)
            id2_list.append(id2)
        if similarities:
            probs = softmax(similarities, temperature=0.2)
            max_idx = np.argmax(probs)
            if probs[max_idx] > 0.5:
                best_matches_1to2[id1] = id2_list[max_idx]

    best_matches_2to1 = {}
    for id2, data2 in person_db["cam2"].items():
        up2 = get_hist_average(data2["up_histories"])
        low2 = get_hist_average(data2["low_histories"])
        if up2 is None and low2 is None:
            continue
        similarities = []
        id1_list = []
        for id1, data1 in person_db["cam1"].items():
            up1 = get_hist_average(data1["up_histories"])
            low1 = get_hist_average(data1["low_histories"])
            if up1 is None and low1 is None:
                continue
            sim_list = []
            if up1 is not None and up2 is not None:
                sim_list.append(compare_histograms(up1, up2))
            if low1 is not None and low2 is not None:
                sim_list.append(compare_histograms(low1, low2))
            if len(sim_list) == 0:
                continue
            similarity = np.mean(sim_list)
            similarities.append(similarity)
            id1_list.append(id1)
        if similarities:
            probs = softmax(similarities, temperature=0.2)
            max_idx = np.argmax(probs)
            if probs[max_idx] > 0.5:
                best_matches_2to1[id2] = id1_list[max_idx]

    matched_pairs = []
    for id1, id2 in best_matches_1to2.items():
        if id2 in best_matches_2to1 and best_matches_2to1[id2] == id1:
            matched_pairs.append((id1, id2))

    # ----------- 매칭 결과 표시 -----------

    for id1, id2 in matched_pairs:
        pos1 = person_db["cam1"][id1]["last_position"]
        pos2 = person_db["cam2"][id2]["last_position"]
        if pos1 is not None:
            cv2.putText(
                frames["cam1"],
                f"↔ ID {id2}",
                (pos1[0], pos1[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        if pos2 is not None:
            cv2.putText(
                frames["cam2"],
                f"↔ ID {id1}",
                (pos2[0], pos2[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    for cam_name in frames:
        cv2.imshow(f"Tracking - {cam_name}", frames[cam_name])

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
