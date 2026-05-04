# [프로그램 5-1] YOLOv8 모델 불러오기
# YOLO 설치
# pip install ultralytics
import torch

from ultralytics import YOLO


# 장치 선택
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 사전 학습된 모델 로드
model = YOLO("yolov8n.pt").to(device)

# [프로그램 5-2] 테스트 이미지 불러오기

#from PIL import Image

#img = Image.open("sample.jpg")     # RGB 이미지

#COCO 이미지 예시를 사용
from PIL import Image
from io import BytesIO
import requests

url = "http://farm9.staticflickr.com/8261/8702818172_05105ccf66_z.jpg"

resp = requests.get(url, timeout=10)
resp.raise_for_status()  # HTTP 오류면 예외

img = Image.open(BytesIO(resp.content)).convert("RGB")

# [프로그램 5-3] 모델 추론 실행하기
results = model.predict(img, device=device)

# [프로그램 5-4] 탐지 결과를 텍스트로 확인하기
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls)
        score = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()   # 경계 상자 좌표
        print(f"class={cls_id}, score={score:.3f}, box=({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})")

# [프로그램 5-5] 탐지 결과를 시각화하기
results[0].save("detected.jpg")      # 탐지 결과 이미지 저장
results[0].show()                    # 주피터/콜랩 환경에서 시각화

