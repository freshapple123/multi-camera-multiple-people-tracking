import os
import cv2
import numpy as np
from ultralytics import YOLO

# OpenMP 오류 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# HS 히스토그램 추출 함수
def extract_hs_histogram_from_cropped(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# 히스토그램 유사도 비교 함수
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# EMA 적용 함수
def get_hist_average(histories, alpha=0.2):
    if not histories:
        return None
    smoothed_hist = histories[0]
    for hist in histories[1:]:
        smoothed_hist = alpha * hist + (1 - alpha) * smoothed_hist
    return smoothed_hist


# 크롭 및 히스토그램 추출
def get_cropped_and_histogram(kp_array, frame, person_idx, indices):
    valid_points = []
    for idx in indices:
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
        center_x, center_y = int(np.mean(valid_points[:, 0])), int(
            np.mean(valid_points[:, 1])
        )
        return cropped_img, hist, (center_x, center_y)
    return None, None, None


# 비디오 선택
a = 1
video_path = {
    1: r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\1st_angle\1st_angle.mp4",
    2: r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\2nd_angle\2nd_angle.mp4",
    3: r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\3rd_angle\3rd_angle.mp4",
    4: r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\4th_angle\4th_angle.mp4",
    5: r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\5th_angle\5th_angle.mp4",
}[a]

cap = cv2.VideoCapture(video_path)

model = YOLO("yolov8n-pose.pt")
upper_indices = [5, 6, 11, 12]
lower_indices = [13, 14, 15, 16]
full_indices = list(set(upper_indices + lower_indices))

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
        # 각각 상체, 하체, 전신
        _, upper_hist, center_pos = get_cropped_and_histogram(
            kp_array, frame, person_idx, upper_indices
        )
        _, lower_hist, _ = get_cropped_and_histogram(
            kp_array, frame, person_idx, lower_indices
        )
        _, full_hist, _ = get_cropped_and_histogram(
            kp_array, frame, person_idx, full_indices
        )

        if upper_hist is None or lower_hist is None or full_hist is None:
            continue

        matched_id = None
        max_similarity = -1

        for pid, data in person_db.items():
            upper_avg = get_hist_average(data["upper"], alpha=0.2)
            lower_avg = get_hist_average(data["lower"], alpha=0.2)
            full_avg = get_hist_average(data["full"], alpha=0.2)
            if upper_avg is None or lower_avg is None or full_avg is None:
                continue

            sim_upper = compare_histograms(upper_hist, upper_avg)
            sim_lower = compare_histograms(lower_hist, lower_avg)
            sim_full = compare_histograms(full_hist, full_avg)

            total_sim = 0.4 * sim_upper + 0.4 * sim_lower + 0.2 * sim_full

            if total_sim > max_similarity:
                matched_id = pid
                max_similarity = total_sim

        if matched_id is None or max_similarity < 0.3:
            matched_id = next_id
            next_id += 1
            person_db[matched_id] = {
                "upper": [],
                "lower": [],
                "full": [],
                "last_position": center_pos,
                "frames_since_seen": 0,
            }

        person_db[matched_id]["upper"].append(upper_hist)
        person_db[matched_id]["lower"].append(lower_hist)
        person_db[matched_id]["full"].append(full_hist)

        for key in ["upper", "lower", "full"]:
            if len(person_db[matched_id][key]) > 10:
                person_db[matched_id][key].pop(0)

        person_db[matched_id]["last_position"] = center_pos
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

    for pid in list(person_db.keys()):
        if pid not in current_ids:
            person_db[pid]["frames_since_seen"] += 1
            if person_db[pid]["frames_since_seen"] > 300:
                del person_db[pid]

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
