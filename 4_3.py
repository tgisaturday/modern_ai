# [프로그램 4-1] MNIST의 축소 버전 구성하기
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=transform,
    download=True
)

subset = torch.utils.data.Subset(train_dataset, range(5000))
loader = DataLoader(subset, batch_size=32, shuffle=True)

# [프로그램 4-2] CNN 모델 구성하기
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # → (B, 8, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)                           # → (B, 8, 14, 14)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # → (B, 16, 14, 14)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SmallCNN()

# [프로그램 4-3] 손실 함수와 최적화 알고리즘 설정하기
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# [프로그램 4-4] 학습 루프 실행하기
num_epochs = 3

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for images, labels in loader:
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# [프로그램 4-5] 학습 모델 시각화하기
import matplotlib.pyplot as plt

model.eval()

with torch.no_grad():
    sample_images, sample_labels = next(iter(loader))
    preds = model(sample_images).argmax(dim=1)

plt.figure(figsize=(8, 4))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(sample_images[i].squeeze(), cmap='gray')
    plt.title(f"pred: {preds[i].item()}")
    plt.axis('off')

plt.show()
