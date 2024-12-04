import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import heapq  # For prioritized replay
import os
import argparse

# Initialize Pygame
pygame.init()

# Game constants
WINDOW_SIZE = 400
GRID_SIZE = 20
GRID_WIDTH = WINDOW_SIZE // GRID_SIZE
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class AISnake(nn.Module):
    def __init__(self):
        super(AISnake, self).__init__()
        # CNN for spatial pattern recognition
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 channels: snake, food, walls
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Direct feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(36, 128),  # 28 original + 4 relative food dir + 4 tail dir
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(64 * GRID_WIDTH * GRID_WIDTH + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256)
        )
        
        # Value and Advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
        # Planning network for future moves
        self.planning_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # 4 potential future moves * 4 steps ahead
        )
        
    def forward(self, spatial_x, feature_x):
        # Process spatial and feature inputs separately
        spatial_features = self.conv_net(spatial_x)
        direct_features = self.feature_net(feature_x)
        
        # Combine features
        combined = torch.cat([spatial_features, direct_features], dim=1)
        combined = self.combined(combined)
        
        # Get value and advantage
        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        
        # Get future move planning
        planning = self.planning_net(combined)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values, planning

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        
    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.memory) >= self.capacity:
            # Remove lowest priority item
            idx = self.priorities.index(min(self.priorities))
            self.memory.pop(idx)
            self.priorities.pop(idx)
            
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        
    def sample(self, batch_size, alpha=0.6):
        if len(self.memory) == 0:
            return [], [], []
            
        # Convert priorities to probabilities
        probs = np.array(self.priorities) ** alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), min(batch_size, len(self.memory)), p=probs)
        samples = [self.memory[idx] for idx in indices]
        weights = (len(self.memory) * probs[indices]) ** (-0.4)
        weights /= weights.max()
        
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant to ensure non-zero

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Snake - AI Learning (Advanced Version)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.prev_distance = 0
        self.visited_positions = set()
        self.curriculum_level = 0
        self.debug_info = {}
        self.snake = None
        self.direction = None
        self.food = None
        self.score = 0
        self.steps = 0
        self.max_steps = 0
        self.performance_history = deque(maxlen=100)  # Track recent performance
        self.impossible_moves = set()  # Track impossible moves
        self.reset_game()
        
    def reset_game(self):
        self.snake = [(GRID_WIDTH//2, GRID_WIDTH//2)]
        self.direction = random.randint(0, 3)
        self.score = 0
        self.steps = 0
        self.max_steps = 50 * len(self.snake)
        self.prev_distance = 0
        self.visited_positions = {self.snake[0]}
        self.curriculum_level = min(self.score // 2, 3)  # Advance curriculum every 2 points
        self.food = self.spawn_food()
        self.prev_distance = self.get_food_distance()
        self.performance_history.clear()
        return self.get_state()

    def spawn_food(self):
        if self.curriculum_level == 0:
            # Spawn food closer to snake in early curriculum
            head = self.snake[0]
            while True:
                x = head[0] + random.randint(-5, 5)
                y = head[1] + random.randint(-5, 5)
                x = max(0, min(x, GRID_WIDTH-1))
                y = max(0, min(y, GRID_WIDTH-1))
                if (x, y) not in self.snake:
                    return (x, y)
        else:
            while True:
                food = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_WIDTH-1))
                if food not in self.snake:
                    return food

    def get_food_distance(self):
        head = self.snake[0]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

    def get_state(self):
        # 1. Create spatial representation (grid)
        spatial_state = np.zeros((3, GRID_WIDTH, GRID_WIDTH))  # 3 channels: snake, food, walls
        
        # Fill snake positions
        for i, segment in enumerate(self.snake):
            spatial_state[0, segment[1], segment[0]] = 1.0 if i == 0 else 0.5  # Head is 1.0, body is 0.5
        
        # Fill food position
        spatial_state[1, self.food[1], self.food[0]] = 1.0
        
        # Fill wall information (edges)
        spatial_state[2, 0, :] = 1.0  # Top wall
        spatial_state[2, -1, :] = 1.0  # Bottom wall
        spatial_state[2, :, 0] = 1.0  # Left wall
        spatial_state[2, :, -1] = 1.0  # Right wall
        
        # 2. Create feature vector
        head = self.snake[0]
        tail = self.snake[-1]
        feature_state = []
        
        # Original 8-directional vision
        directions = [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]
        for dx, dy in directions:
            wall_dist = 0
            food_dist = 0
            body_dist = 0
            x, y = head
            found_food = False
            found_body = False
            
            while 0 <= x < GRID_WIDTH and 0 <= y < GRID_WIDTH:
                x += dx
                y += dy
                wall_dist += 1
                
                if not found_food and (x, y) == self.food:
                    food_dist = wall_dist
                    found_food = True
                
                if not found_body and (x, y) in self.snake:
                    body_dist = wall_dist
                    found_body = True
            
            feature_state.extend([wall_dist/GRID_WIDTH, 
                                food_dist/GRID_WIDTH if found_food else 0,
                                body_dist/GRID_WIDTH if found_body else 0])
        
        # Add relative food direction
        food_dir = [0, 0, 0, 0]  # UP, RIGHT, DOWN, LEFT
        dx = self.food[0] - head[0]
        dy = self.food[1] - head[1]
        if abs(dx) > abs(dy):
            food_dir[RIGHT if dx > 0 else LEFT] = 1
        else:
            food_dir[DOWN if dy > 0 else UP] = 1
        feature_state.extend(food_dir)
        
        # Add tail direction
        tail_dir = [0, 0, 0, 0]  # UP, RIGHT, DOWN, LEFT
        dx = tail[0] - head[0]
        dy = tail[1] - head[1]
        if abs(dx) > abs(dy):
            tail_dir[RIGHT if dx > 0 else LEFT] = 1
        else:
            tail_dir[DOWN if dy > 0 else UP] = 1
        feature_state.extend(tail_dir)
        
        # Current direction one-hot encoding
        for i in range(4):
            feature_state.append(1.0 if i == self.direction else 0.0)
        
        return torch.FloatTensor(spatial_state), torch.FloatTensor(feature_state)

    def is_move_possible(self, new_head):
        # Check if move would result in immediate death
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_WIDTH or
            new_head in self.snake[:-1]):  # Exclude tail as it will move
            return False
        return True

    def get_possible_moves(self):
        possible = set()
        head = self.snake[0]
        moves = {
            UP: (head[0], head[1] - 1),
            RIGHT: (head[0] + 1, head[1]),
            DOWN: (head[0], head[1] + 1),
            LEFT: (head[0] - 1, head[1])
        }
        for direction, new_head in moves.items():
            if self.is_move_possible(new_head):
                possible.add(direction)
        return possible

    def adjust_difficulty(self):
        # Calculate average recent performance
        avg_score = sum(self.performance_history) / len(self.performance_history) if self.performance_history else 0
        
        # Adjust max steps based on performance and snake length
        base_steps = 50 * len(self.snake)
        if avg_score < 5:  # Struggling
            self.max_steps = int(base_steps * 1.5)  # Give more time
        elif avg_score > 10:  # Doing well
            self.max_steps = int(base_steps * 0.8)  # Make it more challenging
        else:
            self.max_steps = base_steps

        # Adjust curriculum level based on performance
        if avg_score > 15:
            self.curriculum_level = min(3, self.curriculum_level + 1)
        elif avg_score < 3:
            self.curriculum_level = max(0, self.curriculum_level - 1)

    def step(self, action):
        self.steps += 1
        self.direction = action
        
        # Get new head position
        head = self.snake[0]
        if self.direction == UP:
            new_head = (head[0], head[1] - 1)
        elif self.direction == RIGHT:
            new_head = (head[0] + 1, head[1])
        elif self.direction == DOWN:
            new_head = (head[0], head[1] + 1)
        else:  # LEFT
            new_head = (head[0] - 1, head[1])
        
        # Update impossible moves
        self.impossible_moves = set(range(4)) - self.get_possible_moves()
        
        # Check if game over
        if not self.is_move_possible(new_head) or self.steps >= self.max_steps:
            # Store performance before reset
            self.performance_history.append(self.score)
            return self.get_state(), -3.0, True
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Multi-objective rewards
        reward = 0
        
        # Distance reward (increased from 0.1 to 0.3)
        new_distance = self.get_food_distance()
        distance_reward = (self.prev_distance - new_distance) * 0.3
        self.prev_distance = new_distance
        reward += distance_reward
        
        # Survival reward (scaled by available moves)
        possible_moves = len(self.get_possible_moves())
        reward += 0.01 * possible_moves  # More reward for maintaining options
        
        # Exploration reward
        if new_head not in self.visited_positions:
            reward += 0.1
        else:
            reward -= 0.2  # Penalty for revisiting
        self.visited_positions.add(new_head)
        
        # Food reward with curriculum
        if new_head == self.food:
            self.score += 1
            self.performance_history.append(self.score)
            reward += 3.0 + self.curriculum_level  # More reward at higher levels
            self.food = self.spawn_food()
            self.adjust_difficulty()  # Adjust difficulty after success
            self.prev_distance = self.get_food_distance()
            self.visited_positions = {pos for pos in self.snake}
            return self.get_state(), reward, False
        else:
            self.snake.pop()
            
        # Adjust difficulty periodically
        if self.steps % 100 == 0:
            self.adjust_difficulty()
            
        return self.get_state(), reward, False

    def render(self):
        self.screen.fill(BLACK)
        
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN,
                           (segment[0]*GRID_SIZE, segment[1]*GRID_SIZE,
                            GRID_SIZE-2, GRID_SIZE-2))
        
        # Draw food
        pygame.draw.rect(self.screen, RED,
                        (self.food[0]*GRID_SIZE, self.food[1]*GRID_SIZE,
                         GRID_SIZE-2, GRID_SIZE-2))
        
        # Draw debug information
        y_offset = 10
        for key, value in self.debug_info.items():
            if isinstance(value, (int, float)):
                text = self.font.render(f"{key}: {value:.2f}", True, WHITE)
            else:
                text = self.font.render(f"{key}: {value}", True, WHITE)
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
        
        # Draw curriculum level
        level_text = self.font.render(f"Level: {self.curriculum_level}", True, YELLOW)
        self.screen.blit(level_text, (WINDOW_SIZE - 100, 10))
        
        pygame.display.flip()
        self.clock.tick(FPS)

def test_model(infinite_mode=True):
    pygame.init()
    game = SnakeGame()
    ai = AISnake()
    ai.eval()
    torch.set_grad_enabled(False)
    
    scores = []
    max_score = 0
    games_played = 0
    total_score = 0
    
    print("Testing AI Snake... Press Q to quit, T to toggle infinite mode")
    running = True
    
    while running:
        state = game.reset_game()
        if not infinite_mode:
            game.max_steps = 50 * len(game.snake)
        else:
            game.max_steps = float('inf')
        
        spatial_state = torch.FloatTensor(state[0]).unsqueeze(0)
        feature_state = torch.FloatTensor(state[1]).unsqueeze(0)
        done = False
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_t:
                        infinite_mode = not infinite_mode
                        print(f"Infinite mode: {'ON' if infinite_mode else 'OFF'}")
                        if infinite_mode:
                            game.max_steps = float('inf')
                        else:
                            game.max_steps = 50 * len(game.snake)
            
            # Get action from model
            q_values, planning = ai(spatial_state, feature_state)
            
            # Smart default behavior when untrained
            head = game.snake[0]
            food = game.food
            dx = food[0] - head[0]
            dy = food[1] - head[1]
            
            # Check for immediate collisions
            next_positions = {
                UP: (head[0], head[1] - 1),
                RIGHT: (head[0] + 1, head[1]),
                DOWN: (head[0], head[1] + 1),
                LEFT: (head[0] - 1, head[1])
            }
            
            # Filter out moves that would cause immediate death
            safe_moves = []
            for move, pos in next_positions.items():
                if (0 <= pos[0] < GRID_SIZE and 
                    0 <= pos[1] < GRID_SIZE and 
                    pos not in game.snake[:-1]):  # Exclude tail as it will move
                    safe_moves.append(move)
            
            if not safe_moves:  # If no safe moves, use model output
                action = q_values.squeeze().argmax().item()
            else:
                # Prioritize moving towards food when possible
                if abs(dx) > abs(dy):
                    if dx > 0 and RIGHT in safe_moves:
                        action = RIGHT
                    elif dx < 0 and LEFT in safe_moves:
                        action = LEFT
                    else:
                        if dy > 0 and DOWN in safe_moves:
                            action = DOWN
                        elif dy < 0 and UP in safe_moves:
                            action = UP
                        else:
                            action = random.choice(safe_moves)
                else:
                    if dy > 0 and DOWN in safe_moves:
                        action = DOWN
                    elif dy < 0 and UP in safe_moves:
                        action = UP
                    else:
                        if dx > 0 and RIGHT in safe_moves:
                            action = RIGHT
                        elif dx < 0 and LEFT in safe_moves:
                            action = LEFT
                        else:
                            action = random.choice(safe_moves)
            
            # Take action
            next_state, reward, done = game.step(action)
            
            # Update state tensors
            spatial_state = torch.FloatTensor(next_state[0]).unsqueeze(0)
            feature_state = torch.FloatTensor(next_state[1]).unsqueeze(0)
            
            # Update debug info
            avg_score = total_score / max(1, games_played)
            game.debug_info = {
                'Q-Value': q_values.mean().item(),
                'Planning': planning.mean().item(),
                'Score': game.score,
                'Max Score': max_score,
                'Games Played': games_played,
                'Avg Score': avg_score,
                'Mode': 'Infinite' if infinite_mode else 'Normal',
                'Action': ['UP', 'RIGHT', 'DOWN', 'LEFT'][action],
                'Safe Moves': len(safe_moves)
            }
            
            game.render()
            pygame.time.delay(50)
        
        # Update statistics
        scores.append(game.score)
        max_score = max(max_score, game.score)
        games_played += 1
        total_score += game.score
        
        # Plot progress
        plt.clf()
        plt.plot(scores)
        plt.axhline(y=total_score / games_played, color='r', linestyle='--', label='Average')
        plt.title(f'Snake Game Test Scores (Max: {max_score})')
        plt.xlabel('Game')
        plt.ylabel('Score')
        plt.legend()
        plt.pause(0.1)
    
    pygame.quit()
    
    # Final statistics
    print(f"\nTesting Results:")
    print(f"Games Played: {games_played}")
    print(f"Max Score: {max_score}")
    print(f"Average Score: {total_score / games_played:.2f}")
    
    # Plot final results
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.axhline(y=total_score / games_played, color='r', linestyle='--', label='Average')
    plt.title('Snake Game Test Scores')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

def train_snake_ai(episodes=1000):
    game = SnakeGame()
    ai = AISnake()
    memory = PrioritizedReplayBuffer(5000)
    optimizer = optim.Adam(ai.parameters(), lr=0.001)
    
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.97
    batch_size = 64
    gamma = 0.98
    
    scores = []
    episode = 0
    best_score = 0
    
    print("Training AI Snake... Press Q to quit")
    running = True
    
    while running and episode < episodes:
        state = game.reset_game()
        spatial_state = torch.FloatTensor(state[0]).unsqueeze(0)
        feature_state = torch.FloatTensor(state[1]).unsqueeze(0)
        total_reward = 0
        done = False
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
            
            # Get Q-values and select action
            ai.train()
            q_values, planning = ai(spatial_state, feature_state)
            
            # Select action
            if random.random() < epsilon:
                if random.random() < 0.7:  # 70% chance to move towards food
                    head = game.snake[0]
                    dx = game.food[0] - head[0]
                    dy = game.food[1] - head[1]
                    if abs(dx) > abs(dy):
                        action = RIGHT if dx > 0 else LEFT
                    else:
                        action = DOWN if dy > 0 else UP
                else:
                    action = random.randint(0, 3)
            else:
                action = q_values.squeeze().argmax().item()
            
            # Take action
            next_state, reward, done = game.step(action)
            next_spatial = torch.FloatTensor(next_state[0]).unsqueeze(0)
            next_feature = torch.FloatTensor(next_state[1]).unsqueeze(0)
            
            # Store transition
            memory.push(
                (spatial_state.squeeze(0).numpy(), feature_state.squeeze(0).numpy()),
                action,
                reward,
                (next_spatial.squeeze(0).numpy(), next_feature.squeeze(0).numpy()),
                done
            )
            
            if len(memory.memory) >= batch_size:
                batch, indices, weights = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Convert states to proper format
                states_spatial = torch.stack([torch.FloatTensor(s[0]) for s in states])
                states_feature = torch.stack([torch.FloatTensor(s[1]) for s in states])
                next_states_spatial = torch.stack([torch.FloatTensor(s[0]) for s in next_states])
                next_states_feature = torch.stack([torch.FloatTensor(s[1]) for s in next_states])
                
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)
                weights = torch.tensor(weights, dtype=torch.float32)
                
                # Compute Q values
                current_q, _ = ai(states_spatial, states_feature)
                current_q = current_q.gather(1, actions.unsqueeze(1))
                
                with torch.no_grad():
                    next_q, _ = ai(next_states_spatial, next_states_feature)
                    next_q = next_q.max(1)[0]
                target_q = rewards + gamma * next_q * (1 - dones)
                
                # Compute loss with importance sampling weights
                loss = (weights * (current_q.squeeze() - target_q) ** 2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update priorities
                priorities = abs(current_q.squeeze().detach() - target_q).numpy()
                memory.update_priorities(indices, priorities)
            
            # Update states
            spatial_state = next_spatial
            feature_state = next_feature
            total_reward += reward
            
            # Update debug info
            game.debug_info = {
                'Q-Value': q_values.mean().item(),
                'Planning': planning.mean().item(),
                'Epsilon': epsilon,
                'Episode': episode,
                'Score': game.score,
                'Action': ['UP', 'RIGHT', 'DOWN', 'LEFT'][action]
            }
            
            game.render()
        
        # Update statistics
        scores.append(game.score)
        if game.score > best_score:
            best_score = game.score
        
        # Plot progress
        if episode % 10 == 0:
            plt.clf()
            plt.plot(scores)
            plt.axhline(y=sum(scores) / len(scores), color='r', linestyle='--', label='Average')
            plt.title(f'Training Scores (Best: {best_score})')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend()
            plt.pause(0.1)
        
        # Decay epsilon
        epsilon = max(0.02, epsilon * 0.97)
        episode += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test the Snake AI')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode to run the AI in')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_snake_ai()
    else:
        test_model()
