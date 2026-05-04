# [프로그램 6-1] 데이터로더 구성하기
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),     # (H, W, C) → (C, H, W), 값 범위 [0,1]
    transforms.Normalize((0.5,), (0.5,))    # [0,1] → [-1,1]로 정규화
])

dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=transform,
    download=True
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

# [프로그램 6-2] 생성자 네트워크 구성하기
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            # 입력: (B, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1, padding=0),  # → (B,128,7,7)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),          # → (B,64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),            # → (B,1,28,28)
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

generator = Generator()

# [프로그램 6-3] 판별자 네트워크 구성하기
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 입력: (B,1,28,28)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),     # → (B,64,14,14)
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # → (B,128,7,7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),                                             # → (B,128*7*7)
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

discriminator = Discriminator()

# [프로그램 6-4] 이진 교차 엔트로피 손실과 Adam 최적화 기법을 설정하기
criterion = nn.BCELoss()

optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

latent_dim = 64

# [프로그램 6-5] 이진 교차 엔트로피 손실을 이용한 GAN 학습 과정
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator.to(device)
discriminator.to(device)

for epoch in range(num_epochs):
    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device)      # (B,1,28,28)
        batch_size = real_imgs.size(0)

        # 1. 판별자 학습 단계
        optimizer_D.zero_grad()

        # 진짜 이미지에 대한 판별 결과와 손실
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        d_real = discriminator(real_imgs)
        loss_real = criterion(d_real, real_labels)

        # 생성된 가짜 이미지에 대한 판별 결과와 손실
        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_imgs = generator(z)
        d_fake = discriminator(fake_imgs.detach())
        loss_fake = criterion(d_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # 2. 생성자 학습 단계
        optimizer_G.zero_grad()

        d_fake = discriminator(fake_imgs)
        loss_G = criterion(d_fake, real_labels)   # 생성자는 가짜를 진짜처럼 보이게 만들고자 함
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}: D={loss_D.item():.4f}, G={loss_G.item():.4f}")

# [프로그램 6-6] 학습된 생성자를 이용해 이미지 샘플 생성 및 시각화하기
import matplotlib.pyplot as plt

generator.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim, 1, 1).to(device)
    samples = generator(z).cpu()

plt.figure(figsize=(6, 4))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(samples[i].squeeze(), cmap='gray')
    plt.axis("off")
plt.show()
