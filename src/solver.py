import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import warnings
from src.game import Game2048
import cProfile
import pstats

warnings.filterwarnings("ignore", category=RuntimeWarning)


class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DQNSolver2048:
    def __init__(self, n_games=1000, gamma=0.99, lr=1e-3, batch_size=8192, memory_size=5e6, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, target_update=10):
        self.n_games = n_games
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=int(memory_size))
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.actions = ['up', 'down', 'left', 'right']
        self.scores, self.moves, self.max_tiles = [], [], []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim, self.output_dim = 18, 4
        self.policy_net = DQNNet(self.input_dim, self.output_dim).to(self.device)
        self.target_net = DQNNet(self.input_dim, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.steps = 0

    def get_state(self, board):
        board_log2 = np.where(board > 0, np.log2(board), 0).flatten().astype(np.float32)
        empty_tiles = np.count_nonzero(board == 0) / 16.0
        max_tile = np.max(board)
        max_tile_log2 = np.log2(max_tile) / 16.0 if max_tile > 0 else 0.0
        features = np.concatenate([board_log2, [empty_tiles, max_tile_log2]]).astype(np.float32)
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def get_valid_moves(self, game):
        valid_moves = []
        original_board = game.board.copy()
        for move in self.actions:
            test_game = Game2048()
            test_game.board = original_board.copy()
            try:
                test_game.move(move)
                if not np.array_equal(test_game.board, original_board):
                    valid_moves.append(move)
            except Exception:
                continue
        return valid_moves

    def compute_reward(self, score_before, score_after, max_tile_before, board_after):
        reward = score_after - score_before
        max_tile_after = np.max(board_after)
        if max_tile_after > max_tile_before:
            reward += (max_tile_after ** 1.5 - max_tile_before ** 1.5)
        if max_tile_after >= 4096 and max_tile_before < 4096:
            reward += 2000
        elif max_tile_after >= 2048 and max_tile_before < 2048:
            reward += 500
        elif max_tile_after >= 1024 and max_tile_before < 1024:
            reward += 100
        reward -= 1
        return reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_moves)
        with torch.no_grad():
            q_values = self.policy_net(state.unsqueeze(0)).detach().cpu().numpy()[0]
        move_indices = [self.actions.index(m) for m in valid_moves]
        valid_qs = [q_values[i] for i in move_indices]
        return valid_moves[np.argmax(valid_qs)]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor([self.actions.index(a) for a in actions], device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        import time
        total_start = time.time()
        game_times = []
        for game_idx in range(self.n_games):
            game_start = time.time()
            game = Game2048()
            move_count = 0
            state = self.get_state(game.board)
            while not game.is_game_over() and move_count < 1000:
                valid_moves = self.get_valid_moves(game)
                if not valid_moves:
                    break
                action = self.act(state, valid_moves)
                board_before = game.board.copy()
                score_before = game.score
                max_tile_before = np.max(board_before)
                game.move(action)
                reward = self.compute_reward(score_before, game.score, max_tile_before, game.board)
                next_state = self.get_state(game.board)
                done = game.is_game_over()
                if done:
                    reward -= 100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                move_count += 1
                self.steps += 1
                if self.steps % 4 == 0:
                    self.replay()
                if self.steps % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            self.scores.append(game.score)
            self.moves.append(move_count)
            self.max_tiles.append(np.max(game.board))
            game_end = time.time()
            game_times.append(game_end - game_start)
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if (game_idx + 1) % 100 == 0:
                print(f"Game {game_idx+1}/{self.n_games} | Avg score: {np.mean(self.scores[-100:]):.2f} | Epsilon: {self.epsilon:.3f} | Avg game time: {np.mean(game_times[-100:]):.3f}s")
        total_end = time.time()
        print(f"DQN agent played {self.n_games} games.")
        print(f"Average score: {np.mean(self.scores):.2f}")
        print(f"Average moves: {np.mean(self.moves):.2f}")
        print(f"Average highest tile: {np.mean(self.max_tiles):.2f}")
        print(f"Max tile achieved: {np.max(self.max_tiles)}")
        print(f"Total training time: {total_end - total_start:.2f} seconds")
        print(f"Average time per game: {np.mean(game_times):.3f} seconds")

    def plot_results(self):
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.hist(self.scores, bins=30, color='skyblue', edgecolor='black')
        plt.title('Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.subplot(1, 3, 2)
        plt.hist(self.moves, bins=30, color='salmon', edgecolor='black')
        plt.title('Move Count Distribution')
        plt.xlabel('Moves')
        plt.ylabel('Frequency')
        plt.subplot(1, 3, 3)
        max_tile = int(np.max(self.max_tiles))
        tile_bins = [2 ** i for i in range(1, int(np.log2(max_tile)) + 1)]
        if 0 in self.max_tiles:
            tile_bins = [0] + tile_bins
        counts = [self.max_tiles.count(tile) for tile in tile_bins]
        x_pos = np.arange(len(tile_bins))
        bars = plt.bar(x_pos, counts, color='limegreen', edgecolor='black', width=0.8)
        plt.title('Highest Tile Distribution')
        plt.xlabel('Highest Tile')
        plt.ylabel('Frequency')
        plt.xticks(x_pos, labels=tile_bins, rotation=45)
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count),
                     ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    agent = DQNSolver2048(n_games=10000)
    cProfile.run('agent.train()', 'profile_stats')
    agent.plot_results()
    print("Profiling results saved to 'profile_stats'. To view, run:")
    print("    python -m pstats profile_stats")
