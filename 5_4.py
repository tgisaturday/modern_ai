# [프로그램 5-6] 임의의 도형 분할 데이터 생성하기
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import random

# 간단한 도형 이미지를 생성하는 함수
def generate_shape_image(size=64):
    img = Image.new("L", (size, size), 0)  # 흑백, 배경 0
    draw = ImageDraw.Draw(img)
    shape_type = random.choice(["circle", "square"])
    x1, y1 = random.randint(5, 25), random.randint(5, 25)
    x2, y2 = x1 + random.randint(20, 35), y1 + random.randint(20, 35)
    
    if shape_type == "circle":
        draw.ellipse((x1, y1, x2, y2), fill=255)
    else:
        draw.rectangle((x1, y1, x2, y2), fill=255)
        
    return np.array(img, dtype=np.float32) / 255.0

# Dataset 정의
class ShapeDataset(Dataset):
    def __init__(self, n=200):
        self.images = [generate_shape_image() for _ in range(n)]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        # 입력과 정답 동일 (self-supervised segmentation)
        return torch.tensor(img).unsqueeze(0), torch.tensor(img)

dataset = ShapeDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# [프로그램 5-7] 간단한 U-Net 모델 구성
import torch.nn as nn
import torch.nn.functional as F

class UNetMini(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 인코더
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU()
        )  # output: (B,8,64,64)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(8,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1), nn.ReLU()
        )  # output: (B,16,32,32)
        
        # 디코더
        self.dec1 = nn.Sequential(
            nn.Conv2d(24,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1), nn.ReLU()
        )  # output: (B,16,64,64)
        
        self.final = nn.Conv2d(16, 1, 1)  # output: (B,1,64,64)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # 인코더
        x1 = self.enc1(x)               # (B,8,64,64)
        x2 = self.pool(x1)              # (B,8,32,32)
        x2 = self.enc2(x2)              # (B,16,32,32)
        
        # 업샘플링 + 잔차 연결
        x3 = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)  # (B,16,64,64)
        x_cat = torch.cat([x1, x3], dim=1)   # (B, 8+16 = 24, 64,64)
        
        # 디코더
        x4 = self.dec1(x_cat)           # (B,16,64,64)
        out = torch.sigmoid(self.final(x4))  # (B,1,64,64)
        
        return out

model = UNetMini()

# [프로그램 5-8] BCE 손실 함수 사용
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# [프로그램 5-9] 학습 루프 실행하기
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0.0
    for images, masks in loader:
        # images shape: (8,1,64,64)
        # masks  shape: (8,64,64)
        
        optimizer.zero_grad()
        preds = model(images).squeeze(1)  # (8,64,64)
        
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# [프로그램 5-10] 분할 결과 시각화하기
model.eval()
sample_img, sample_mask = dataset[0]
with torch.no_grad():
    pred = model(sample_img.unsqueeze(0)).squeeze().numpy()

plt.figure(figsize=(8,3))
plt.subplot(1,3,1); plt.imshow(sample_img.squeeze(), cmap='gray'); plt.title("Input"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(sample_mask, cmap='gray'); plt.title("Target"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(pred, cmap='gray'); plt.title("Prediction"); plt.axis('off')
plt.show()
