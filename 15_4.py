# [프로그램 15-1] 오목 환경 구성하기
import numpy as np

BOARD_SIZE = 5
WIN_COND = 3

class GomokuEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)  # 0: empty, 1: black, -1: white
        self.current_player = 1
        return self.board.copy()

    def available_actions(self):
        return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                if self.board[r, c] == 0]

    def step(self, action):
        r, c = action
        self.board[r, c] = self.current_player

        if self.check_win(self.current_player):
            return self.board.copy(), 1, True

        if len(self.available_actions()) == 0:
            return self.board.copy(), 0, True

        self.current_player *= -1
        return self.board.copy(), 0, False

    def check_win(self, player):
        board = self.board
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if c + WIN_COND <= BOARD_SIZE and all(board[r, c+i] == player for i in range(WIN_COND)):
                    return True
                if r + WIN_COND <= BOARD_SIZE and all(board[r+i, c] == player for i in range(WIN_COND)):
                    return True
                if r + WIN_COND <= BOARD_SIZE and c + WIN_COND <= BOARD_SIZE and \
                   all(board[r+i, c+i] == player for i in range(WIN_COND)):
                    return True
                if r + WIN_COND <= BOARD_SIZE and c - WIN_COND >= -1 and \
                   all(board[r+i, c-i] == player for i in range(WIN_COND)):
                    return True
        return False

# [프로그램 15-2] 정책-가치 신경망 정의하기 
import torch
import torch.nn as nn

class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.policy_head = nn.Linear(64, BOARD_SIZE * BOARD_SIZE)
        self.value_head  = nn.Linear(64, 1)

    def forward(self, x):
        feats = self.features(x)
        policy_logits = self.policy_head(feats)   # (B,25)
        value = torch.tanh(self.value_head(feats))  # (B,1)
        return policy_logits, value

# [프로그램 15-3] PUCT 기반 MCTS 구현하기

#노드 구조
class Node:
    def __init__(self, state, player):
        self.state = state
        self.player = player
        self.children = {}
        self.N = 0    # 방문 횟수
        self.W = 0    # 누적 가치
        self.Q = 0    # 평균 가치
        self.P = None # 사전 확률

#선택 단계: PUCT 규칙 적용
def select_child(node, c_puct=1.0):
    best_score = -1e9
    best_action, best_child = None, None

    for action, child in node.children.items():
        U = c_puct * child.P * np.sqrt(node.N + 1) / (child.N + 1)
        score = child.Q + U
        if score > best_score:
            best_score, best_child, best_action = score, child, action

    return best_action, best_child

#확장 및 평가 단계
def expand_and_evaluate(node, model):
    state_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)
    policy_logits, value = model(state_tensor)

    policy = torch.softmax(policy_logits, dim=-1).detach().cpu().numpy().flatten()
    actions = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]
    total_valid = sum(node.state[r, c] == 0 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))

    for idx, action in enumerate(actions):
        if node.state[action[0], action[1]] == 0:
            node.children[action] = Node(node.state.copy(), -node.player)
            node.children[action].P = policy[idx] / max(total_valid, 1)

    return float(value.item())

#역전파 단계
def backpropagate(path, value):
    for node in reversed(path):
        node.N += 1
        node.W += value
        node.Q = node.W / node.N
        value = -value

#전체 MCTS 실행:
def mcts_search(root, model, n_sim=50):
    for _ in range(n_sim):
        node = root
        path = [node]

        while node.children:
            action, node = select_child(node)
            path.append(node)

        value = expand_and_evaluate(node, model)
        backpropagate(path, value)

    actions, visits = zip(*[(a, child.N) for a, child in root.children.items()])
    visits = np.array(visits, dtype=float)
    probs = visits / visits.sum()
    return actions, probs

# [프로그램 15-4-1] AI vs AI 자동 플레이 구현하기 
def render_board(board):
    symbols = {0: ".", 1: "X", -1: "O"}
    for r in range(BOARD_SIZE):
        print(" ".join(symbols[board[r, c]] for c in range(BOARD_SIZE)))
    print()

env = GomokuEnv()
model = PolicyValueNet()

state = env.reset()
done = False
turn = 0

print("=== AI vs AI 자동 플레이 ===")

while not done and turn < 10:
    print(f"\nTurn {turn}, Player {'X' if env.current_player == 1 else 'O'}")
    render_board(state)

    root = Node(state, env.current_player)
    actions, probs = mcts_search(root, model, n_sim=50)

    # 확률 보드 시각화
    prob_board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for (r, c), p in zip(actions, probs):
        prob_board[r, c] = p

    np.set_printoptions(precision=2, suppress=True)
    print("Action probability board:")
    print(prob_board)

    # greedy 선택
    action = actions[np.argmax(probs)]
    print("선택된 행동:", action)

    state, reward, done = env.step(action)
    turn += 1

print("\n최종 보드:")
render_board(state)

# [프로그램 15-4-2] 사람 vs AI 자동 플레이 구현하기 
env = GomokuEnv()
model = PolicyValueNet()

state = env.reset()
done = False

print("=== 사람(X) vs AI(O) ===")
print("좌표는 (row, col), 0부터 시작")

while not done:
    render_board(state)

    if env.current_player == 1:
        # 사람 차례
        move = input("당신의 수 (예: 2 3): ")
        r, c = map(int, move.split())
        state, reward, done = env.step((r, c))
    else:
        # AI 차례
        root = Node(state, env.current_player)
        actions, probs = mcts_search(root, model, n_sim=50)

        prob_board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for (r, c), p in zip(actions, probs):
            prob_board[r, c] = p

        print("AI action probability board:")
        np.set_printoptions(precision=2, suppress=True)
        print(prob_board)

        action = actions[np.argmax(probs)]
        print("AI 선택:", action)

        state, reward, done = env.step(action)

print("\n게임 종료")
render_board(state)