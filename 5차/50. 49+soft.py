import os
import cv2
from ultralytics import YOLO
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_angle_path(n):
    angle_str = (
        f"{n}st" if n == 1 else f"{n}nd" if n == 2 else f"{n}rd" if n == 3 else f"{n}th"
    )
    return rf"D:\\REid\\data\\retail\\MMPTracking_training\\To_seperate_for_video\\{angle_str}_angle\\{angle_str}_angle.mp4"


model = YOLO("yolov8n-pose.pt")

upper_indices = [5, 6, 11, 12]
lower_indices = [11, 12, 15, 16]

a = 1
b = 5

video_paths = {"cam1": get_angle_path(a), "cam2": get_angle_path(b)}
caps = {
    "cam1": cv2.VideoCapture(video_paths["cam1"]),
    "cam2": cv2.VideoCapture(video_paths["cam2"]),
}

person_db = {}
next_id = 0
stable_match_counter = {}
stable_threshold = 5

print("▶ 멀티카메라 영상 분석 시작... 'Q' 키로 종료")

while True:
    frames = {}
    results = {}
    detected_people = {"cam1": [], "cam2": []}

    for cam_name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue
        frames[cam_name] = frame
        results[cam_name] = model(frame)[0].keypoints

    if not frames:
        break

    for cam_name, keypoints in results.items():
        if keypoints is None or keypoints.data.shape[0] == 0:
            continue

        kp_array = keypoints.data.cpu().numpy()
        frame = frames[cam_name]

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

            detected_people[cam_name].append(
                {
                    "up_hist": up_hist,
                    "low_hist": low_hist,
                    "center": center_pos,
                    "frame": frame,
                    "camera": cam_name,
                }
            )

    cam1_people = detected_people["cam1"]
    cam2_people = detected_people["cam2"]

    similarity_matrix = np.zeros((len(cam1_people), len(cam2_people)))

    for i, p1 in enumerate(cam1_people):
        for j, p2 in enumerate(cam2_people):
            sim_list = []
            if p1["up_hist"] is not None and p2["up_hist"] is not None:
                sim_list.append(compare_histograms(p1["up_hist"], p2["up_hist"]))
            if p1["low_hist"] is not None and p2["low_hist"] is not None:
                sim_list.append(compare_histograms(p1["low_hist"], p2["low_hist"]))
            if len(sim_list) > 0:
                similarity_matrix[i, j] = np.mean(sim_list)

    softmax_row = np.apply_along_axis(softmax, 1, similarity_matrix)
    softmax_col = np.apply_along_axis(softmax, 0, similarity_matrix)

    matched_pairs = []
    for i in range(len(cam1_people)):
        j_best = np.argmax(softmax_row[i])
        i_back = np.argmax(softmax_col[:, j_best])
        if i == i_back and similarity_matrix[i, j_best] > 0.3:
            matched_pairs.append((i, j_best))

    assigned_ids = {}
    for i, j in matched_pairs:
        p1 = cam1_people[i]
        p2 = cam2_people[j]

        assigned_id = None
        for pid, data in person_db.items():
            avg_up = get_hist_average(data["up_histories"])
            avg_low = get_hist_average(data["low_histories"])
            sim_list = []
            if p1["up_hist"] is not None and avg_up is not None:
                sim_list.append(compare_histograms(p1["up_hist"], avg_up))
            if p1["low_hist"] is not None and avg_low is not None:
                sim_list.append(compare_histograms(p1["low_hist"], avg_low))
            if len(sim_list) > 0 and np.mean(sim_list) > 0.3:
                assigned_id = pid
                break

        if assigned_id is None:
            assigned_id = next_id
            person_db[assigned_id] = {
                "up_histories": [],
                "low_histories": [],
                "last_position": p1["center"],
            }
            next_id += 1

        for person in [p1, p2]:
            if person["up_hist"] is not None:
                person_db[assigned_id]["up_histories"].append(person["up_hist"])
                if len(person_db[assigned_id]["up_histories"]) > 10:
                    person_db[assigned_id]["up_histories"].pop(0)
            if person["low_hist"] is not None:
                person_db[assigned_id]["low_histories"].append(person["low_hist"])
                if len(person_db[assigned_id]["low_histories"]) > 10:
                    person_db[assigned_id]["low_histories"].pop(0)
            if person["center"] is not None:
                cv2.putText(
                    person["frame"],
                    f"ID:{assigned_id}",
                    person["center"],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    for cam_name in frames:
        cv2.imshow(f"Tracking - {cam_name}", frames[cam_name])

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
