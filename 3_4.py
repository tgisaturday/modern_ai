# [프로그램 3-1] 데이터셋과 데이터로더 구성
import torch
from torch.utils.data import Dataset, DataLoader

# 간단한 데이터셋 정의
class SimpleDataset(Dataset):
    def __init__(self):
        torch.manual_seed(0)
        self.X = torch.randn(200, 2)                # 입력 데이터 (200 × 2)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).long()  # 이진 라벨 생성

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SimpleDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# [프로그램 3-2] 모델, 손실 함수, 최적화 알고리즘 정의
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# [프로그램 3-3] 이진 분류 모델의 학습 루프 구현
num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    
    for X_batch, y_batch in loader:
        optimizer.zero_grad()                      # 1) 기울기 초기화
        outputs = model(X_batch).squeeze()         # 2) 순전파 수행
        loss = criterion(outputs, y_batch.float()) # 3) 손실 계산
        loss.backward()                            # 4) 기울기 계산
        optimizer.step()                           # 5) 파라미터 업데이트
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")

# [프로그램 3-4] 학습된 모델을 간단히 평가하기
with torch.no_grad():
    X_all = dataset.X
    y_all = dataset.y
    
    preds = (model(X_all).squeeze() > 0.5).long()
    accuracy = (preds == y_all).float().mean().item()

print("Training accuracy:", accuracy)

