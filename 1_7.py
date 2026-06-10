# [프로그램 1-3] 넘파이를 이용한 선형 연산
import numpy as np

# 입력 벡터
x = np.array([1.0, 2.0, 3.0])      # shape: (3,)

# 가중치 벡터
w = np.array([0.5, 0.2, -0.3])     # shape: (3,)

# 편향
b = 0.1                             # shape: scalar

# 선형 결합
y = x @ w + b                       # shape: scalar
print("NumPy output:", y)

# [프로그램 1-4] 파이토치를 활용한 선형 연산
import torch

# 입력 텐서
x = torch.tensor([1.0, 2.0, 3.0])        # shape: (3,)

# 가중치 텐서 (기울기 추적)
w = torch.tensor([0.5, 0.2, -0.3], requires_grad=True)   # shape: (3,)

# 편향
b = torch.tensor(0.1, requires_grad=True)                # shape: ()

# 선형 결합
y = x @ w + b                                            # shape: ()
print("PyTorch output:", y.item())

# 자동 미분
y.backward()
print("grad w:", w.grad)                                 # shape: (3,)
print("grad b:", b.grad)                                 # shape: ()

# [프로그램 1-5] 파이토치를 활용한 선형 연산
import numpy as np
import torch

# NumPy 배열
A = np.array([[1., 2., 3.],
              [4., 5., 6.]])               # shape: (2,3)

b = np.array([1., 0., -1.])                # shape: (3,)

# 브로드캐스팅 적용
C = A + b                                   # shape: (2,3)
print("NumPy broadcast result:\n", C)

# PyTorch 텐서
A_t = torch.tensor([[1., 2., 3.],
                    [4., 5., 6.]])          # shape: (2,3)

b_t = torch.tensor([1., 0., -1.])           # shape: (3,)

C_t = A_t + b_t                              # shape: (2,3)
print("PyTorch broadcast result:\n", C_t)

# [프로그램 1-6] 간단한 비선형 함수 적용
import torch

# 입력 텐서
x = torch.tensor([-1.0, 0.0, 1.0, 2.0])     # shape: (4,)

# ReLU 적용
y = torch.relu(x)                           # shape: (4,)
print("ReLU output:", y)

# [프로그램 1-7] 연산 장치(CPU, GPU) 자동 선택 구조
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

x = torch.tensor([1.0, 2.0, 3.0], device=device)  # shape: (3,)
w = torch.tensor([0.5, 0.2, -0.3], device=device, requires_grad=True)
b = torch.tensor(0.1, device=device, requires_grad=True)

y = x @ w + b                                     # shape: ()
print("result:", y.item())

# [프로그램 1-8] 맷플롯립을 활용한 연산 결과 시각화
import torch
import matplotlib.pyplot as plt

# 입력 값 생성
x = torch.linspace(-3, 3, steps=100)      # shape: (100,)

# 선형 변환 (가중치와 편향)
w = 1.0
b = 0.5
y_linear = w * x + b

# 비선형 함수 적용 (ReLU)
y_relu = torch.relu(y_linear)

# 시각화
plt.figure(figsize=(6, 4))
plt.plot(x.numpy(), y_linear.numpy(), label="Linear output")
plt.plot(x.numpy(), y_relu.numpy(), label="ReLU output")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)

plt.xlabel("Input x")
plt.ylabel("Output")
plt.title("Linear Transformation and ReLU Activation")
plt.legend()
plt.grid(True)
plt.show()