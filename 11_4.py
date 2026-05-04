# [프로그램 11-1] CartPole 환경 생성하고 초기화하기
#!pip install gymnasium

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode=None)

state, info = env.reset()
print("Initial state:", state)    # shape: (4,)

# [프로그램 11-2] 무작위 행동으로 환경과 상호작용하기
total_reward = 0.0

for step in range(10):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)

    print(f"step {step}")
    print(" action:", action)
    print(" next_state:", next_state)
    print(" reward:", reward)
    print(" terminated:", terminated)
    print(" truncated:", truncated)
    print("--------------")

    total_reward += reward

    if terminated or truncated:
        break

# [프로그램 11-3] 한 에피소드 전체 실행하기
state, info = env.reset()
total_reward = 0.0
done = False

while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    state = next_state

    if terminated or truncated:
        done = True

print("Episode reward:", total_reward)

