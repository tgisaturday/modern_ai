# [프로그램 11-4] 환경 및 라이브러리 준비하기
#!pip install gymnasium torch numpy

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [프로그램 11-5] Q-네트워크 정의하기
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

policy_net = QNetwork().to(device)
target_net = QNetwork().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# [프로그램 11-6] 경험 재생 버퍼 구성하기
buffer = deque(maxlen=50000)

def add_experience(state, action, reward, next_state, done):
    buffer.append((state, action, reward, next_state, done))

# [프로그램 11-7]  ε-탐욕 행동 선택 함수 구성하기
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0,1)
    else:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = policy_net(state)
        return q_values.argmax(dim=1).item()

# [프로그램 11-8] DQN 업데이트 함수 구성하기
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
gamma = 0.99

def train_step(batch_size=32):
    if len(buffer) < batch_size:
        return

    minibatch = random.sample(buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.tensor(
        states, dtype=torch.float32, device=device
    )
    actions = torch.tensor(
        actions, dtype=torch.int64, device=device
    )
    rewards = torch.tensor(
        rewards, dtype=torch.float32, device=device
    )
    next_states = torch.tensor(
        next_states, dtype=torch.float32, device=device
    )
    dones = torch.tensor(
        dones, dtype=torch.float32, device=device
    )

    q_values = policy_net(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]
        target = rewards + gamma * next_q * (1.0 - dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# [프로그램 11-9] 학습 루프 구성하기
env = gym.make("CartPole-v1")

num_episodes = 300
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

for ep in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    
    while True:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        add_experience(state, action, reward, next_state, done)
        
        train_step()
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if ep % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Episode {ep} | Reward: {total_reward}")

# [프로그램 11-10] 학습된 정책 확인하기
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = select_action(state, epsilon=0.0)
    next_state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    state = next_state
    done = terminated or truncated

print("Test episode reward:", total_reward)

