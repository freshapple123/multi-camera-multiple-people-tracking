import os
import cv2
from ultralytics import YOLO
import numpy as np

# OpenMP 중복 허용 에러 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 🔹 HS 히스토그램 추출 함수
def extract_hs_histogram_from_cropped(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# 🔹 히스토그램 비교 함수
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# 🔹 사람 크롭 + 히스토그램 추출 함수
def get_cropped_and_histogram(kp_array, frame, person_idx, desired_indices):
    valid_points = []

    for idx in desired_indices:
        x, y, conf = kp_array[person_idx][idx]
        if conf >= 0.5:
            valid_points.append([x, y])

    valid_points = np.array(valid_points)

    if len(valid_points) >= 3:
        x_min = int(np.min(valid_points[:, 0]))
        x_max = int(np.max(valid_points[:, 0]))
        y_min = int(np.min(valid_points[:, 1]))
        y_max = int(np.max(valid_points[:, 1]))

        h, w, _ = frame.shape
        x_min = max(x_min, 0)
        x_max = min(x_max, w)
        y_min = max(y_min, 0)
        y_max = min(y_max, h)

        cropped_img = frame[y_min:y_max, x_min:x_max]
        hist = extract_hs_histogram_from_cropped(cropped_img)

        center_x = int(np.mean(valid_points[:, 0]))
        center_y = int(np.mean(valid_points[:, 1]))

        return cropped_img, hist, (center_x, center_y)

    return None, None, None


# 🔹 영상 경로
video_path = r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\1st_angle\1st_angle.mp4"
cap = cv2.VideoCapture(video_path)

# 🔹 YOLOv8 Pose 모델
model = YOLO("yolov8n-pose.pt")

# 🔹 키포인트 인덱스 (어깨/골반)
desired_indices = [5, 6, 11, 12]

# 🔹 사람 ID 관리
person_id_counter = 0
previous_people = []

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
    current_people = []

    for person_idx in range(len(kp_array)):
        cropped_img, hist, center_pos = get_cropped_and_histogram(
            kp_array, frame, person_idx, desired_indices
        )

        if hist is None:
            continue

        matched_id = None
        max_similarity = -1

        for prev in previous_people:
            similarity = compare_histograms(hist, prev["hist"])
            if similarity > 0.7 and similarity > max_similarity:
                matched_id = prev["id"]
                max_similarity = similarity

        if matched_id is None:
            matched_id = person_id_counter
            person_id_counter += 1

        current_people.append({"id": matched_id, "hist": hist})

        # 🔹 ID 표시
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

    previous_people = current_people

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
