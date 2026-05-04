# [프로그램 2-3] 단일 뉴런의 학습 과정 관찰하기
import torch
import torch.nn as nn
import torch.optim as optim

# 입력과 정답
# shape: x (1,), y (1,)
x = torch.tensor([2.0])
y = torch.tensor([4.0])

# 단일 뉴런 모델: y_pred = w * x + b
w = torch.tensor([0.1], requires_grad=True)   # shape: (1,)
b = torch.tensor([0.0], requires_grad=True)   # shape: (1,)

optimizer = optim.SGD([w, b], lr=0.1)
criterion = nn.MSELoss()

# 학습 반복
for step in range(10):
    optimizer.zero_grad()
    
    y_pred = w * x + b             # shape: (1,)
    loss = criterion(y_pred, y)    # shape: ()
    loss.backward()
    
    print(f"step {step} | w={w.item():.4f} | b={b.item():.4f} | grad_w={w.grad.item():.4f} | loss={loss.item():.4f}")
    
    optimizer.step()

# [프로그램 2-4] 학습률 변화가 가중치 이동에 미치는 영향 비교하기
import torch
import torch.nn as nn
import torch.optim as optim

def run_lr(rate):
    w = torch.tensor([0.1], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    optimizer = optim.SGD([w, b], lr=rate)
    x = torch.tensor([2.0])
    y = torch.tensor([4.0])
    criterion = nn.MSELoss()
    
    history = []
    for _ in range(8):
        optimizer.zero_grad()
        y_pred = w * x + b
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        history.append((w.item(), loss.item()))
    return history

# 두 학습률 비교
lr_small = run_lr(0.01)
lr_large = run_lr(0.5)

print("Small LR (0.01):", lr_small)
print("Large LR (0.5):", lr_large)

# [프로그램 2-5] 손실 표면을 2차원으로 시각화하기
import numpy as np
import matplotlib.pyplot as plt

# 손실 계산 함수
def loss_surface(w):
    x = 2.0
    y = 4.0
    return (w * x - y) ** 2

# w 구간 설정
ws = np.linspace(-2, 4, 200)
losses = [loss_surface(w) for w in ws]

plt.plot(ws, losses)
plt.title("Loss Surface for w")
plt.xlabel("w")
plt.ylabel("Loss")
plt.show()

