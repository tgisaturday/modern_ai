# !pip install gymnasium torch numpy

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# [프로그램 12-7] 정책 네트워크와 가치 네트워크 정의하기
# 정책: state -> action probabilities
class PolicyNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # probs 출력
        )

    def forward(self, x):
        return self.net(x)


# 가치: state -> V(s)
class ValueNet(nn.Module):
    def __init__(self, state_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


policy = PolicyNet().to(device)
value = ValueNet().to(device)

policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
value_opt = optim.Adam(value.parameters(), lr=1e-3)


# [프로그램 12-8] 롤아웃 수집 함수 구성하기
def collect_trajectories(env, policy, batch_size=2048):
    states, actions, rewards, dones, log_probs = [], [], [], [], []

    obs, _ = env.reset()
    steps = 0

    while steps < batch_size:
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = policy(s).squeeze(0)  # [action_dim]
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            logp = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = bool(terminated or truncated)

        states.append(obs)
        actions.append(action.item())
        rewards.append(float(reward))
        dones.append(done)
        log_probs.append(float(logp.item()))

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

        steps += 1

    return states, actions, rewards, dones, log_probs


# [프로그램 12-9] GAE(λ)와 누적 보상 계산
def compute_gae(states, rewards, dones, value_net, gamma=0.99, lam=0.95):
    states_t = torch.tensor(states, dtype=torch.float32).to(device)

    with torch.no_grad():
        values = value_net(states_t).view(-1).cpu().numpy()  # [T]

    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)

    last_gae = 0.0
    next_value = 0.0  # done이면 0으로 bootstrap

    for t in reversed(range(T)):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae

        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]

        next_value = values[t]

    return advantages, returns


# [프로그램 12-10] PPO 클리핑 목적 함수로 정책 업데이트하기
def ppo_update(
    policy, value_net,
    states, actions, old_log_probs,
    advantages, returns,
    clip_eps=0.2, K_epochs=4, batch_size=64,
    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5
):
    states_t = torch.tensor(states, dtype=torch.float32).to(device)
    actions_t = torch.tensor(actions, dtype=torch.int64).to(device)
    old_logp_t = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
    adv_t = torch.tensor(advantages, dtype=torch.float32).to(device)
    ret_t = torch.tensor(returns, dtype=torch.float32).to(device)

    N = states_t.size(0)

    # 전체 rollout 기준 advantage 정규화 (PPO 안정화에 매우 중요)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    for _ in range(K_epochs):
        # minibatch shuffle
        idxs = torch.randperm(N, device=device)

        for start in range(0, N, batch_size):
            mb = idxs[start:start + batch_size]

            s_batch = states_t[mb]
            a_batch = actions_t[mb]
            old_log_batch = old_logp_t[mb]
            adv_batch = adv_t[mb]
            ret_batch = ret_t[mb]

            probs = policy(s_batch)
            dist = torch.distributions.Categorical(probs=probs)
            new_log = dist.log_prob(a_batch)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log - old_log_batch)

            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            values_pred = value_net(s_batch).view(-1)
            value_loss = (values_pred - ret_batch).pow(2).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            policy_opt.zero_grad()
            value_opt.zero_grad()

            loss.backward()

            # 안정화: gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)

            policy_opt.step()
            value_opt.step()


# [프로그램 12-11] 학습 루프 구성하기
env = gym.make("CartPole-v1")

for iteration in range(50):
    states, actions, rewards, dones, log_probs = collect_trajectories(env, policy, batch_size=2048)
    advantages, returns = compute_gae(states, rewards, dones, value, gamma=0.99, lam=0.95)

    ppo_update(
        policy, value,
        states, actions, log_probs,
        advantages, returns,
        clip_eps=0.2, K_epochs=4, batch_size=64,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5
    )

    if (iteration + 1) % 5 == 0:
        print(f"Iteration {iteration + 1}/50 completed.")


env.close()


# [프로그램 12-12] 학습된 정책 실행하기
def evaluate_policy(env_id, policy, device, episodes=10, greedy=True, render=False):
    policy.eval()
    env_kwargs = {"render_mode": "human"} if render else {}
    env = gym.make(env_id, **env_kwargs)

    scores = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0

        while not done:
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = policy(s)
                if greedy:
                    action = probs.argmax(dim=1).item()
                else:
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample().item()

            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)

        scores.append(total)
        print(f"Test episode {ep + 1}/{episodes} reward: {total}")

    env.close()

    avg = sum(scores) / len(scores)
    print(f"Average reward over {episodes} episodes: {avg}")
    print(f"Min/Max reward: {min(scores)} / {max(scores)}")

    return avg, scores


avg, rewards = evaluate_policy("CartPole-v1", policy, device, episodes=5, greedy=True, render=False)