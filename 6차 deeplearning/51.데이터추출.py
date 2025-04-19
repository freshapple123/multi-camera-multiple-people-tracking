import os
import cv2
from ultralytics import YOLO
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def extract_valid_crop(points, frame):
    if len(points) < 3:
        return None
    x_min, x_max = int(np.min(points[:, 0])), int(np.max(points[:, 0]))
    y_min, y_max = int(np.min(points[:, 1])), int(np.max(points[:, 1]))
    h, w, _ = frame.shape
    x_min, x_max = max(x_min, 0), min(x_max, w)
    y_min, y_max = max(y_min, 0), min(y_max, h)
    return frame[y_min:y_max, x_min:x_max]


def get_valid_points(kp_array, person_idx, indices, threshold=0.5):
    points = []
    for idx in indices:
        x, y, conf = kp_array[person_idx][idx]
        if conf >= threshold:
            points.append([x, y])
    return np.array(points)


def get_angle_path(n):
    angle_str = (
        f"{n}st" if n == 1 else f"{n}nd" if n == 2 else f"{n}rd" if n == 3 else f"{n}th"
    )
    return rf"D:\\REid\\data\\retail\\MMPTracking_training\\To_seperate_for_video\\{angle_str}_angle\\{angle_str}_angle.mp4"


# 카메라 번호 지정
angle = 1  # 예: 1~6 중 선택
video_path = get_angle_path(angle)

# 저장할 폴더 경로
save_base = rf"D:\\REid\\cropped_dataset\\cam{angle}"
os.makedirs(os.path.join(save_base, "upper"), exist_ok=True)
os.makedirs(os.path.join(save_base, "lower"), exist_ok=True)

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(video_path)
upper_indices = [5, 6, 11, 12]
lower_indices = [11, 12, 15, 16]

frame_idx = 0
image_count = 0

print("▶ Crop 이미지 추출 시작...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0].keypoints
    if results is None or results.data.shape[0] == 0:
        frame_idx += 1
        continue

    kp_array = results.data.cpu().numpy()

    for person_idx in range(len(kp_array)):
        up_points = get_valid_points(kp_array, person_idx, upper_indices)
        low_points = get_valid_points(kp_array, person_idx, lower_indices)

        upper_crop = extract_valid_crop(up_points, frame)
        lower_crop = extract_valid_crop(low_points, frame)

        if upper_crop is not None:
            up_path = os.path.join(
                save_base, "upper", f"person{image_count:04d}_f{frame_idx}.jpg"
            )
            cv2.imwrite(up_path, upper_crop)

        if lower_crop is not None:
            low_path = os.path.join(
                save_base, "lower", f"person{image_count:04d}_f{frame_idx}.jpg"
            )
            cv2.imwrite(low_path, lower_crop)

        image_count += 1

    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ 총 {image_count}개 상/하체 crop 저장 완료")
