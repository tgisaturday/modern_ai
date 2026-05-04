#[프로그램 1-2] VS Code에서 첫 번째 프로그램 작성하기
import torch
import numpy as np

a = np.array([1, 2, 3])
print(a * 2)

x = torch.tensor([1.0, 2.0, 3.0])
y = x * 2
print(y)

print(torch.__version__)
print(torch.cuda.is_available())
