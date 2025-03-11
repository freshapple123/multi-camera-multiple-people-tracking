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
    return preprocess(image).unsqueeze(0)


# 마스크 생성
def create_mask(output):
    output_predictions = output["out"].argmax(1).squeeze().detach().cpu().numpy()
    mask = (output_predictions == 15).astype(np.uint8)  # 15: 'person' 클래스
    return mask


# 이미지 로드 및 추론
image_path = "C:/Users/Ok/Desktop/a.png"
input_image = preprocess(image_path)
output = model(input_image)

# 마스크 생성
mask = create_mask(output)

# 원본 이미지 불러오기
original_image = cv2.imread(image_path)

# 마스크 크기 맞추기
mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

# 마스크를 컬러맵으로 변환 (원하는 경우)
colored_mask = cv2.applyColorMap(mask_resized * 255, cv2.COLORMAP_JET)

# 세그멘테이션된 이미지 합성
segmented_image = cv2.addWeighted(original_image, 0.6, colored_mask, 0.4, 0)

# 마스크에 맞춰서 원본 이미지에서 오리기
masked_image = cv2.bitwise_and(original_image, original_image, mask=mask_resized)

# 결과 출력
cv2.imshow("Segmented Clothes", segmented_image)
cv2.imshow("Masked Image", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
