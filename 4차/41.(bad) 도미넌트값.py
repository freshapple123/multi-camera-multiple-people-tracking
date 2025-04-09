import os
import cv2
from ultralytics import YOLO
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------- 함수 정의 --------------------


# 히스토그램에서 상위 k개의 도미넌트 값 추출
def get_top_k_dominant_bins(hist, k=4):
    indices = np.argsort(hist)[::-1][:k]
    return hist[indices]


# HS 히스토그램 추출 → 도미넌트 4개만 반환
def extract_hs_histogram_from_cropped(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    flat_hist = hist.flatten()
    return get_top_k_dominant_bins(flat_hist, k=4)


# 도미넌트 벡터 유사도 계산 (cosine similarity)
def compare_histograms(hist1, hist2):
    if np.linalg.norm(hist1) == 0 or np.linalg.norm(hist2) == 0:
        return 0
    return np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))


# EMA (지수 이동 평균)
def get_hist_average(histories, alpha=0.2):
    if len(histories) == 0:
        return None
    smoothed_hist = histories[0]
    for hist in histories[1:]:
        smoothed_hist = alpha * hist + (1 - alpha) * smoothed_hist
    return smoothed_hist


# 키포인트 유효 좌표 추출
def get_valid_points(kp_array, person_idx, indices, threshold=0.5):
    points = []
    for idx in indices:
        x, y, conf = kp_array[person_idx][idx]
        if conf >= threshold:
            points.append([x, y])
    return np.array(points)


# 크롭 영역에서 도미넌트 벡터 및 중심 좌표 추출
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
model = YOLO("yolov8n-pose.pt")

upper_indices = [5, 6, 11, 12]  # 상체
lower_indices = [11, 12, 15, 16]  # 하체

person_db = {}
next_id = 0

print("▶ 영상 분석 시작... 'Q' 키로 종료")

# -------------------- 프레임별 분석 --------------------

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

        for pid, data in person_db.items():
            sim_list = []

            if up_hist is not None and len(data["up_histories"]) > 0:
                avg_up = get_hist_average(data["up_histories"])
                sim_list.append(compare_histograms(up_hist, avg_up))

            if low_hist is not None and len(data["low_histories"]) > 0:
                avg_low = get_hist_average(data["low_histories"])
                sim_list.append(compare_histograms(low_hist, avg_low))

            if len(sim_list) > 0:
                similarity = sum(sim_list) / len(sim_list)

                # 🔽 유사도 출력 로그
                print(f"[Frame] Comparing with ID {pid}: similarity = {similarity:.4f}")

                if similarity > max_similarity:
                    matched_id = pid
                    max_similarity = similarity
        # 🔽 최종 매칭 유사도 확인
        print(f"=> Best match ID: {matched_id} with similarity: {max_similarity:.4f}")

        if matched_id is None or max_similarity < 0.3:
            matched_id = next_id
            next_id += 1
            person_db[matched_id] = {
                "up_histories": [],
                "low_histories": [],
                "last_position": center_pos,
                "frames_since_seen": 0,
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

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
