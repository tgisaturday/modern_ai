# [프로그램 4-6] 시계열 데이터 생성하기
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

t = np.linspace(0, 100, 1000)
data = np.sin(t).astype(np.float32)

window = 20

class SineDataset(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window
    
    def __len__(self):
        return len(self.data) - self.window
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window]      # shape: (20,)
        y = self.data[idx+self.window]          # shape: ()
        return torch.tensor(x).unsqueeze(1), torch.tensor(y)

dataset = SineDataset(data, window)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# [프로그램 4-7] LSTM 모델 정의하기
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)            # out shape: (batch, 20, 16)
        last = out[:, -1, :]             # 마지막 시점의 은닉 상태
        y = self.fc(last)                # shape: (batch, 1)
        return y.squeeze(1)

model = LSTMForecaster()

# [프로그램 4-8] 손실 함수와 최적화 알고리즘 설정하기
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# [프로그램 4-9] 학습 루프 실행하기
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# [프로그램 4-10] 미래 예측 시각화하기
import matplotlib.pyplot as plt

model.eval()

test_input = data[200:200+window]                 # shape: (20,)
current = torch.tensor(test_input).unsqueeze(0).unsqueeze(2)

preds = []

for _ in range(50):
    with torch.no_grad():
        y_hat = model(current)                    # shape: (1,)
    preds.append(y_hat.item())
    
    new_seq = torch.cat([current[:, 1:, :], y_hat.view(1, 1, 1)], dim=1)
    current = new_seq

plt.plot(range(20), test_input, label='Input Sequence')
plt.plot(range(20, 70), preds, label='Predicted')
plt.legend()
plt.title("LSTM Time-Series Forecast")
plt.show()

