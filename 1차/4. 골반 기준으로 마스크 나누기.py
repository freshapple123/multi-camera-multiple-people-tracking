import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# 모델 로드
model = models.segmentation.deeplabv3_resnet101(
    weights="DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1"
)
model.eval()


# 이미지 전처리
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return image, preprocess(image).unsqueeze(0)  # 원본 이미지와 전처리된 이미지 반환


# 마스크 생성
def create_mask(output):
    output_predictions = output["out"].argmax(1).squeeze().detach().cpu().numpy()
    mask = (output_predictions == 15).astype(np.uint8)  # 15: 'person' 클래스
    return mask


# 두 점을 기준으로 마스크를 상의와 하의로 나누기
def split_mask(mask, point1, point2):
    h, w = mask.shape
    upper_mask = np.zeros((h, w), dtype=np.uint8)
    lower_mask = np.zeros((h, w), dtype=np.uint8)

    # 직선 방정식 y = mx + b 계산
    x1, y1 = point1
    x2, y2 = point2
    if x1 != x2:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    else:
        slope = None  # 수직선
        intercept = None

    # 픽셀 위치를 기준으로 상의/하의 나누기
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:  # 사람이 검출된 픽셀만 처리
                if slope is not None:
                    line_y = slope * x + intercept
                    if y < line_y:
                        upper_mask[y, x] = 1
                    else:
                        lower_mask[y, x] = 1
                else:  # 수직선의 경우
                    if x < x1:
                        upper_mask[y, x] = 1
                    else:
                        lower_mask[y, x] = 1

    return upper_mask, lower_mask


# 이미지 로드 및 추론
image_path = "C:/Users/Ok/Desktop/a.png"
original_image, input_image = preprocess(image_path)
output = model(input_image)

# 마스크 생성
mask = create_mask(output)

# 원본 이미지 크기 저장 (PIL.Image 객체에서는 size 사용)
orig_width, orig_height = original_image.size

# 기준 점 정의 (리사이즈 이전 좌표)
point1 = (70, 126)
point2 = (47, 120)

# 리사이즈된 이미지 크기 (520, 520)에 맞게 비례적으로 좌표 변경
resize_height, resize_width = 520, 520

point1_resized = (
    int(point1[0] * resize_width / orig_width),
    int(point1[1] * resize_height / orig_height),
)
point2_resized = (
    int(point2[0] * resize_width / orig_width),
    int(point2[1] * resize_height / orig_height),
)

# 상의와 하의 마스크 분리
upper_mask, lower_mask = split_mask(mask, point1_resized, point2_resized)

# 원본 이미지와 마스크 크기 맞추기 (리사이즈)
upper_colored_mask = cv2.resize(upper_mask, (orig_width, orig_height))
lower_colored_mask = cv2.resize(lower_mask, (orig_width, orig_height))

# 마스크를 컬러맵으로 변환
upper_colored_mask = cv2.applyColorMap(upper_colored_mask * 255, cv2.COLORMAP_JET)
lower_colored_mask = cv2.applyColorMap(lower_colored_mask * 255, cv2.COLORMAP_JET)

# 상의 및 하의 세그멘테이션 합성
upper_segmented = cv2.addWeighted(
    np.array(original_image), 0.6, upper_colored_mask, 0.4, 0
)
lower_segmented = cv2.addWeighted(
    np.array(original_image), 0.6, lower_colored_mask, 0.4, 0
)

# 결과 출력
cv2.imshow("Upper Clothes", upper_segmented)
cv2.imshow("Lower Clothes", lower_segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()
