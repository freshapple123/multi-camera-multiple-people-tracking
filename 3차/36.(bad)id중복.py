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

def get_cropped_and_histogram(kp_array, frame, person_idx, desired_indices):
    valid_points = []
    for idx in desired_indices:
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
        center_x, center_y = int(np.mean(valid_points[:, 0])), int(np.mean(valid_points[:, 1]))
        
        return cropped_img, hist, (center_x, center_y)
    return None, None, None

video_path = r"D:\REid\data\retail\MMPTracking_training\To_seperate_for_video\1st_angle\1st_angle.mp4"
cap = cv2.VideoCapture(video_path)

model = YOLO("yolov8n-pose.pt")
desired_indices = [5, 6, 11, 12]

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
        cropped_img, hist, center_pos = get_cropped_and_histogram(kp_array, frame, person_idx, desired_indices)
        if hist is None:
            continue

        matched_id = None
        max_similarity = -1

        for pid, data in person_db.items():
            if pid in current_ids:
                continue
            avg_hist = get_hist_average(data["histories"], alpha=0.2)
            if avg_hist is not None:
                similarity = compare_histograms(hist, avg_hist)
                if similarity > max_similarity:
                    matched_id = pid
                    max_similarity = similarity

        if matched_id is None or max_similarity < 0.3 or matched_id in current_ids:
            matched_id = next_id
            next_id += 1
            person_db[matched_id] = {"histories": [], "last_position": center_pos}
        
        person_db[matched_id]["histories"].append(hist)
        if len(person_db[matched_id]["histories"]) > 10:
            person_db[matched_id]["histories"].pop(0)

        person_db[matched_id]["last_position"] = center_pos
        current_ids.add(matched_id)
        
        if center_pos is not None:
            cv2.putText(frame, f"ID: {matched_id}", center_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()