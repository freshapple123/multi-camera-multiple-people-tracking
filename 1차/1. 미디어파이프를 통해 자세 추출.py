import cv2
import mediapipe as mp

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 상체 및 하체 랜드마크 정의
UPPER_BODY_LANDMARKS = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
]

MIDDLE_BODY_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
]

LOWER_BODY_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
]

# 이미지 읽기
# image = cv2.imread("C:/Users/Ok/Desktop/a-person/a-person/a.png")
image = cv2.imread("C:/Users/Ok/Desktop/1st_angle/person_0.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mediapipe로 자세 추적
results = pose.process(image_rgb)


# 특정 랜드마크 좌표 출력 함수
def print_landmark_coordinates(landmark_ids, label):
    print(f"\n{label} 랜드마크 좌표:")
    for landmark in landmark_ids:
        point = results.pose_landmarks.landmark[landmark]
        x = int(point.x * image.shape[1])
        y = int(point.y * image.shape[0])
        print(f"{landmark.name}: ({x}, {y})")


# 상체와 하체 랜드마크 그리기
if results.pose_landmarks:
    annotated_image = image.copy()

    # 상체 랜드마크
    for landmark in UPPER_BODY_LANDMARKS:
        point = results.pose_landmarks.landmark[landmark]
        x = int(point.x * image.shape[1])
        y = int(point.y * image.shape[0])
        cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)  # 초록색 원

    # 상하체 기준 랜드마크
    for landmark in MIDDLE_BODY_LANDMARKS:
        point = results.pose_landmarks.landmark[landmark]
        x = int(point.x * image.shape[1])
        y = int(point.y * image.shape[0])
        cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)  # 초록색 원

    # 하체 랜드마크
    for landmark in LOWER_BODY_LANDMARKS:
        point = results.pose_landmarks.landmark[landmark]
        x = int(point.x * image.shape[1])
        y = int(point.y * image.shape[0])
        cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)  # 파란색 원

    # 결과 이미지 저장
    # output_path = "C:/Users/Ok/Desktop/pose_landmark_separation_result.png"
    # cv2.imwrite(output_path, annotated_image)
    # print(f"결과 이미지가 저장되었습니다: {output_path}")

    # 랜드마크 좌표 출력
    print_landmark_coordinates(UPPER_BODY_LANDMARKS, "상체")
    print_landmark_coordinates(MIDDLE_BODY_LANDMARKS, "중간")
    print_landmark_coordinates(LOWER_BODY_LANDMARKS, "하체")

    # 결과 이미지 표시
    cv2.imshow("Upper and Lower Body Separation", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Mediapipe 객체 닫기
pose.close()
