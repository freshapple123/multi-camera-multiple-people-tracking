import os
import shutil

# 원본 파일이 있는 디렉토리와 결과를 저장할 디렉토리 경로
source_dir = "D:/RE-id/data/MMPTracking_training/MMPTracking_training/retail_0"  # 예: "C:/Users/PC/Desktop/images"
destination_dir = (
    "D:/RE-id/data/retail/MMPTracking_training/To_seperate_for_video/6th_angle"
)
# 예: "C:/Users/PC/Desktop/rgb_00001"

# 결과 디렉토리가 없으면 생성
os.makedirs(destination_dir, exist_ok=True)

# source_dir 내의 파일들을 확인
for file_name in os.listdir(source_dir):
    # 파일 이름이 "rgb_"로 시작하고, 중간에 아무 숫자 5자리, 뒤에 "_1"로 끝나는지 확인
    if file_name.startswith("rgb_") and file_name.endswith("_6.jpg"):
        parts = file_name.split("_")
        if len(parts) == 3 and parts[1].isdigit() and len(parts[1]) == 5:
            # 파일 경로 생성
            source_file = os.path.join(source_dir, file_name)
            destination_file = os.path.join(destination_dir, file_name)

            # 파일 복사
            shutil.copy(source_file, destination_file)

print(f"rgb_*****_1 패턴을 가진 파일이 {destination_dir}에 복사되었습니다.")
