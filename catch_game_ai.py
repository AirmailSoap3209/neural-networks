import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
plt.style.use('dark_background')

# Initialize Pygame and Matplotlib
pygame.init()
plt.ion()  # Enable interactive mode

# Game constants
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600
PADDLE_WIDTH = 60
PADDLE_HEIGHT = 10
BALL_SIZE = 10
FPS = 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Neural Network for AI
class AIPlayer(nn.Module):
    def __init__(self):
        super(AIPlayer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
    def forward(self, x):
        # Ensure input is 2D for single samples
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        
    def push(self, state, action, reward, next_state, error=None):
        if error is None:
            error = 1.0  # Max priority for new experiences
        
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
            self.priorities.pop(0)
            
        self.memory.append((state, action, reward, next_state))
        self.priorities.append(error)
    
    def sample(self, batch_size, alpha=0.6):
        if len(self.memory) == 0:
            return None
            
        priorities = np.array(self.priorities) ** alpha
        probabilities = priorities / sum(priorities)
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        
        states, actions, rewards, next_states = zip(*samples)
        return (torch.stack(states), torch.tensor(actions), 
                torch.tensor(rewards), torch.stack(next_states), indices)
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Small constant to ensure non-zero probability
    
    def __len__(self):
        return len(self.memory)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Catch the Ball - AI Learning (Fast Version)")
        self.clock = pygame.time.Clock()
        self.reset_game()
        
    def reset_game(self):
        self.paddle_x = WINDOW_WIDTH // 2
        self.ball_x = random.randint(BALL_SIZE, WINDOW_WIDTH - BALL_SIZE)
        self.ball_y = BALL_SIZE
        self.ball_speed_y = 10  # Increased initial speed
        self.score = 0
        return self.get_state()
    
    def get_state(self):
        # Normalize values between 0 and 1
        state = torch.FloatTensor([
            self.paddle_x / WINDOW_WIDTH,
            self.ball_x / WINDOW_WIDTH,
            self.ball_y / WINDOW_HEIGHT,
            self.ball_speed_y / 10
        ])
        return state
    
    def step(self, action):
        # Move paddle based on action
        if action == 0:  # Left
            self.paddle_x = max(0, self.paddle_x - 5)
        elif action == 2:  # Right
            self.paddle_x = min(WINDOW_WIDTH - PADDLE_WIDTH, self.paddle_x + 5)
        
        # Move ball
        self.ball_y += self.ball_speed_y
        
        # Check if ball is caught
        reward = 0
        done = False
        
        if self.ball_y >= WINDOW_HEIGHT - PADDLE_HEIGHT - BALL_SIZE:
            if self.paddle_x <= self.ball_x <= self.paddle_x + PADDLE_WIDTH:
                reward = 1
                self.score += 1
                # Reset ball
                self.ball_x = random.randint(BALL_SIZE, WINDOW_WIDTH - BALL_SIZE)
                self.ball_y = BALL_SIZE
                self.ball_speed_y = min(self.ball_speed_y + 0.2, 15)  # Increase speed
            else:
                reward = -1
                done = True
        
        return self.get_state(), reward, done
    
    def render(self):
        self.screen.fill(BLACK)
        # Draw paddle
        pygame.draw.rect(self.screen, WHITE, 
                        (self.paddle_x, WINDOW_HEIGHT - PADDLE_HEIGHT, 
                         PADDLE_WIDTH, PADDLE_HEIGHT))
        # Draw ball
        pygame.draw.circle(self.screen, RED, 
                         (int(self.ball_x), int(self.ball_y)), BALL_SIZE)
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(FPS)

class MetricsPlotter:
    def __init__(self):
        self.scores = []
        self.avg_scores = []
        self.episode_rewards = []
        self.epsilon_values = []
        self.loss_values = []
        self.best_avg_score = float('-inf')
        
        # Create figure and subplots
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Training Metrics', fontsize=16)
        
        # Initialize empty lines
        self.score_line, = self.axs[0, 0].plot([], [], 'b-', label='Score')
        self.avg_score_line, = self.axs[0, 0].plot([], [], 'r-', label='Avg Score (100 ep)')
        self.reward_line, = self.axs[0, 1].plot([], [], 'g-', label='Episode Reward')
        self.epsilon_line, = self.axs[1, 0].plot([], [], 'y-', label='Epsilon')
        self.loss_line, = self.axs[1, 1].plot([], [], 'm-', label='Loss')
        
        # Configure subplots
        for ax in self.axs.flat:
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        self.axs[0, 0].set_title('Scores')
        self.axs[0, 1].set_title('Rewards')
        self.axs[1, 0].set_title('Epsilon')
        self.axs[1, 1].set_title('Loss')
        
        plt.tight_layout()
        
    def update(self, score, reward, epsilon, loss):
        # Update data
        self.scores.append(score)
        self.episode_rewards.append(reward)
        self.epsilon_values.append(epsilon)
        if loss is not None:
            self.loss_values.append(loss)
        
        # Calculate running average score
        window_size = min(100, len(self.scores))
        avg_score = np.mean(self.scores[-window_size:])
        self.avg_scores.append(avg_score)
        
        # Update line data
        x = range(len(self.scores))
        self.score_line.set_data(x, self.scores)
        self.avg_score_line.set_data(x, self.avg_scores)
        self.reward_line.set_data(range(len(self.episode_rewards)), self.episode_rewards)
        self.epsilon_line.set_data(range(len(self.epsilon_values)), self.epsilon_values)
        
        if self.loss_values:
            self.loss_line.set_data(range(len(self.loss_values)), self.loss_values)
        
        # Adjust limits for all subplots
        for ax in self.axs.flat:
            ax.relim()
            ax.autoscale_view()
        
        # Draw and flush events
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def train_ai():
    game = Game()
    ai = AIPlayer()
    target_ai = AIPlayer()
    target_ai.load_state_dict(ai.state_dict())
    
    memory = PrioritizedReplayMemory(10000)
    optimizer = optim.Adam(ai.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    criterion = nn.SmoothL1Loss()
    
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    gamma = 0.99
    target_update_freq = 10
    
    # Initialize metrics plotter
    plotter = MetricsPlotter()
    current_reward = 0
    episode_loss = []
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    def save_checkpoint(episode, is_best=False):
        checkpoint = {
            'episode': episode,
            'model_state_dict': ai.state_dict(),
            'target_model_state_dict': target_ai.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epsilon': epsilon,
            'scores': plotter.scores,
            'avg_scores': plotter.avg_scores,
            'episode_rewards': plotter.episode_rewards,
            'best_avg_score': plotter.best_avg_score
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join('models', 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best performance
        if is_best:
            best_model_path = os.path.join('models', 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"\nNew best model saved! Average Score: {plotter.best_avg_score:.2f}")
    
    def try_load_checkpoint():
        checkpoint_path = os.path.join('models', 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print("\nFound existing checkpoint. Do you want to load it? (y/n)")
            response = input().lower()
            if response == 'y':
                checkpoint = torch.load(checkpoint_path)
                ai.load_state_dict(checkpoint['model_state_dict'])
                target_ai.load_state_dict(checkpoint['target_model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                plotter.scores = checkpoint['scores']
                plotter.avg_scores = checkpoint['avg_scores']
                plotter.episode_rewards = checkpoint['episode_rewards']
                plotter.best_avg_score = checkpoint['best_avg_score']
                nonlocal epsilon
                epsilon = checkpoint['epsilon']
                return checkpoint['episode']
        return 0
    
    print("Training AI... Press Q to quit, S to save checkpoint")
    running = True
    episode = try_load_checkpoint()
    
    while running:
        state = game.reset_game()
        done = False
        episode_start_time = pygame.time.get_ticks()
        episode_loss = []
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_s:
                        save_checkpoint(episode)
                        print("\nCheckpoint saved!")
            
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    action = ai(state).argmax().item()
            
            next_state, reward, done = game.step(action)
            time_penalty = (pygame.time.get_ticks() - episode_start_time) / 1000.0
            reward = reward - 0.001 * time_penalty
            current_reward += reward
            
            memory.push(state, action, reward, next_state)
            
            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                if batch is not None:
                    states, actions, rewards, next_states, indices = batch
                    
                    with torch.no_grad():
                        next_q = target_ai(next_states).max(1)[0]
                        target_q = rewards + gamma * next_q * (1 - done)
                    
                    current_q = ai(states).gather(1, actions.unsqueeze(1))
                    loss = criterion(current_q.squeeze(), target_q)
                    episode_loss.append(loss.item())
                    
                    errors = abs(current_q.squeeze().detach() - target_q).numpy()
                    memory.update_priorities(indices, errors)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ai.parameters(), 1.0)
                    optimizer.step()
            
            state = next_state
            game.render()
        
        if episode % target_update_freq == 0:
            target_ai.load_state_dict(ai.state_dict())
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        scheduler.step()
        
        # Update plots with latest metrics
        avg_loss = np.mean(episode_loss) if episode_loss else None
        plotter.update(game.score, current_reward, epsilon, avg_loss)
        
        # Check if this is the best model
        if len(plotter.scores) > 100:
            current_avg_score = np.mean(plotter.scores[-100:])
            if current_avg_score > plotter.best_avg_score:
                plotter.best_avg_score = current_avg_score
                save_checkpoint(episode, is_best=True)
        
        # Auto-save checkpoint every 100 episodes
        if episode % 100 == 0 and episode > 0:
            save_checkpoint(episode)
            print("\nAuto-saved checkpoint!")
        
        current_reward = 0
        episode += 1
        
        if episode % 10 == 0:
            avg_score = np.mean(plotter.scores[-100:]) if len(plotter.scores) > 100 else np.mean(plotter.scores)
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}, Epsilon: {epsilon:.2f}")
    
    # Save final checkpoint before quitting
    save_checkpoint(episode)
    print("\nFinal checkpoint saved!")
    
    pygame.quit()
    plt.ioff()
    plt.close('all')

if __name__ == "__main__":
    train_ai()
