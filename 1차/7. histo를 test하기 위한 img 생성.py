from PIL import Image

# 이미지 크기 설정 (예: 100x100)
width, height = 100, 100

# R=255, G=0, B=50 색으로 이미지 생성
image = Image.new("RGB", (width, height), (255, 0, 50))

# 이미지 저장
image.save("red_blue_image.png")

# 이미지 표시 (선택 사항)
image.show()
