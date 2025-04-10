import os
import cv2
from ultralytics import YOLO
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# -------- HS 히스토그램 추출 --------
def extract_hs_histogram_from_cropped(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# -------- 히스토그램 유사도 비교 --------
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# -------- EMA 평균 --------
def get_hist_average(histories, alpha=0.2):
    if len(histories) == 0:
        return None
    smoothed_hist = histories[0]
    for hist in histories[1:]:
        smoothed_hist = alpha * hist + (1 - alpha) * smoothed_hist
    return smoothed_hist


# -------- 키포인트에서 특정 인덱스 좌표만 추출 --------
def get_valid_points(kp_array, person_idx, indices, threshold=0.5):
    points = []
    for idx in indices:
        x, y, conf = kp_array[person_idx][idx]
        if conf >= threshold:
            points.append([x, y])
    return np.array(points)


# -------- 크롭 후 히스토그램 + 중심 좌표 --------
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


# -------- 모델 및 설정 --------
model = YOLO("yolov8n-pose.pt")

upper_indices = [5, 6, 11, 12]
lower_indices = [11, 12, 15, 16]

# -------- 카메라 파일 경로 설정 --------
a = 1
b = 5


def get_angle_path(n):
    angle_str = (
        f"{n}st" if n == 1 else f"{n}nd" if n == 2 else f"{n}rd" if n == 3 else f"{n}th"
    )
    return rf"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\{angle_str}_angle\{angle_str}_angle.mp4"


video_paths = {"cam1": get_angle_path(a), "cam2": get_angle_path(b)}

caps = {
    "cam1": cv2.VideoCapture(video_paths["cam1"]),
    "cam2": cv2.VideoCapture(video_paths["cam2"]),
}

# -------- 전역 DB --------
person_db = {}
next_id = 0
train_cam = "cam1"  # 이 카메라에서만 학습

print("▶ 멀티카메라 영상 분석 시작... 'Q' 키로 종료")

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

            # -------- 매칭 --------
            matched_id = None
            max_similarity = -1

            for pid, data in person_db.items():
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

            # -------- 학습 카메라일 때만 새로운 ID 생성 --------
            if cam_name == train_cam:
                if matched_id is None or max_similarity < 0.3:
                    matched_id = next_id
                    next_id += 1
                    person_db[matched_id] = {
                        "up_histories": [],
                        "low_histories": [],
                        "last_position": center_pos,
                    }

                if up_hist is not None:
                    person_db[matched_id]["up_histories"].append(up_hist)
                    if len(person_db[matched_id]["up_histories"]) > 10:
                        person_db[matched_id]["up_histories"].pop(0)

                if low_hist is not None:
                    person_db[matched_id]["low_histories"].append(low_hist)
                    if len(person_db[matched_id]["low_histories"]) > 10:
                        person_db[matched_id]["low_histories"].pop(0)

                person_db[matched_id]["last_position"] = center_pos

            # -------- 매칭된 경우에만 ID 표시 --------
            if matched_id == 0 and center_pos is not None:

                if cam_name != train_cam and max_similarity < 0.35:
                    continue
                cv2.putText(
                    frame,
                    f"{cam_name} - ID:{matched_id}",
                    center_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
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
