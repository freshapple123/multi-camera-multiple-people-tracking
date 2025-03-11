import cv2
import numpy as np

# HSV 값 설정 (Hue, Saturation, Value)
hsv_color = np.uint8([[[125, 25, 150]]])  # HSV 값 입력 (0~179, 0~255, 0~255 범위)

# HSV를 BGR로 변환
bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

# 생성된 색상 확인 (BGR 값 출력)
print(f"BGR Color: {bgr_color}")

# 색상 출력
image = np.full((300, 300, 3), bgr_color, dtype=np.uint8)  # 300x300 크기의 이미지 생성
cv2.imshow("Color", image)  # 창에 색상 표시
cv2.waitKey(0)
cv2.destroyAllWindows()
