# [프로그램 12-1] Gymnasium 라이브러리와 파이토치 준비하기
#!pip install gymnasium torch numpy

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [프로그램 12-2] 확률적 정책 네트워크 정의하기
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

policy = PolicyNetwork().to(device)

# [프로그램 12-3] 에피소드 생성: 정책에 따라 궤적 수집하기
def generate_episode(env, policy):
    states, actions, rewards = [], [], []
    state, _ = env.reset()
    done = False
    
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs = policy(state_tensor).squeeze(0)
        
        action = torch.distributions.Categorical(probs).sample().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
        done = terminated or truncated
    
    return states, actions, rewards

# [프로그램 12-4] REINFORCE 업데이트: 누적 보상 기반 정책 경사 적용하기
def compute_returns(rewards, gamma=0.99):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

optimizer = optim.Adam(policy.parameters(), lr=1e-3)

def update_policy(states, actions, returns):
    optimizer.zero_grad()
    
    for s, a, G in zip(states, actions, returns):
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
        probs = policy(s_tensor)
        log_prob = torch.log(probs[0, a])
        
        loss = -log_prob * G    # REINFORCE 손실
        loss.backward()
    
    optimizer.step()

# [프로그램 12-5] 학습 루프 구성하기
env = gym.make("CartPole-v1")
num_episodes = 500

for ep in range(num_episodes):
    states, actions, rewards = generate_episode(env, policy)
    returns = compute_returns(rewards)
    
    update_policy(states, actions, returns)
    
    print(f"Episode {ep}: Reward = {sum(rewards)}")

# [프로그램 12-6] 학습된 정책 실행하기
policy.eval()

state, _ = env.reset()
done = False
total_reward = 0

while not done:
    s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    action = policy(s_tensor).argmax(dim=1).item()
    next_state, reward, terminated, truncated, _ = env.step(action)
    
    total_reward += reward
    state = next_state
    done = terminated or truncated

print("Test episode reward:", total_reward)

