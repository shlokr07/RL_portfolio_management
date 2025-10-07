# src/train_ppo.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Dirichlet
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from rl_data_prep import PortfolioEnv

# ---------- CONFIG ----------
RL_DATASET_PATH = Path("data/processed/rl_dataset.npz")
CHECKPOINT_DIR = Path("models/checkpoints")
RESULTS_DIR = Path("data/results")
INITIAL_CAPITAL = 1_000_000

# IMPROVED PPO Hyperparameters
PPO_CONFIG = {
    "learning_rate": 1e-4,  # Reduced for stability
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.05,  # Increased significantly for exploration
    "max_grad_norm": 0.5,
    "n_epochs": 4,  # Reduced to prevent overfitting
    "batch_size": 128,  # Increased batch size
    "n_training_episodes": 300,
    "eval_frequency": 20,
    "hidden_dims": [256, 128],  # Simpler architecture
    "concentration_scale": 5.0,  # Lower scale for more exploration
    "reward_scale": 100.0,  # Scale rewards for better learning signal
    "use_reward_normalization": True,
    "min_concentration": 0.1,  # Minimum concentration to prevent collapse
}

# ---------- Actor-Critic Network ----------
class PortfolioActorCritic(nn.Module):
    """Improved Actor-Critic with better exploration"""
    
    def __init__(self, n_assets, n_features, hidden_dims=[256, 128], 
                 concentration_scale=5.0, min_concentration=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.n_features = n_features
        self.concentration_scale = concentration_scale
        self.min_concentration = min_concentration
        
        input_dim = n_assets * n_features
        
        # Shared feature extractor with layer norm
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Better than dropout for RL
                nn.Tanh(),  # Smoother than ReLU
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Actor head with residual connection to encourage diversity
        self.actor_hidden = nn.Linear(prev_dim, prev_dim)
        self.actor_output = nn.Linear(prev_dim, n_assets)
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, 1)
        )
        
        # Initialize weights for better exploration
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to encourage exploration"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Actor output gets special initialization for diversity
        nn.init.orthogonal_(self.actor_output.weight, gain=0.01)
        nn.init.constant_(self.actor_output.bias, 1.0)  # Bias towards uniform
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = state.shape[0]
        x = state.reshape(batch_size, -1)
        
        # Shared features
        features = self.shared_net(x)
        
        # Actor with residual-like connection
        actor_hidden = torch.tanh(self.actor_hidden(features))
        actor_logits = self.actor_output(actor_hidden + features)
        
        # Concentration parameters with soft lower bound
        concentrations = F.softplus(actor_logits) * self.concentration_scale
        concentrations = concentrations + self.min_concentration
        
        # Critic
        value = self.critic(features).squeeze(-1)
        
        if single_input:
            concentrations = concentrations.squeeze(0)
            value = value.squeeze(0)
        
        return concentrations, value
    
    def get_action_and_value(self, state, action=None):
        concentrations, value = self.forward(state)
        
        # Add small noise to prevent collapse
        if self.training:
            concentrations = concentrations + torch.randn_like(concentrations) * 0.01
            concentrations = torch.clamp(concentrations, min=self.min_concentration)
        
        dist = Dirichlet(concentrations)
        
        if action is None:
            action = dist.rsample()
        
        # Ensure action is valid
        action = torch.clamp(action, min=1e-6, max=1.0)
        action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value

# ---------- Experience Buffer ----------
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get(self):
        return {
            'states': torch.stack(self.states),
            'actions': torch.stack(self.actions),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'values': torch.stack(self.values),
            'log_probs': torch.stack(self.log_probs),
            'dones': torch.tensor(self.dones, dtype=torch.float32),
        }

# ---------- Running Statistics ----------
class RunningMeanStd:
    """Track running mean and std for normalization"""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = 1
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

# ---------- PPO Trainer ----------
class PPOTrainer:
    def __init__(self, env, config=PPO_CONFIG, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.config = config
        self.device = device
        
        # Initialize network
        self.network = PortfolioActorCritic(
            n_assets=env.n_assets,
            n_features=env.n_features,
            hidden_dims=config['hidden_dims'],
            concentration_scale=config['concentration_scale'],
            min_concentration=config['min_concentration']
        ).to(device)
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.network.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['n_training_episodes'], eta_min=1e-5
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Reward normalization
        self.reward_rms = RunningMeanStd() if config['use_reward_normalization'] else None
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_values = []
        self.eval_results = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        print(f"PPO Trainer initialized on {device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print(f"Configuration: {json.dumps(config, indent=2)}")
    
    def normalize_reward(self, reward):
        """Normalize reward using running statistics"""
        if self.reward_rms is not None:
            self.reward_rms.update(reward)
            return reward / (np.sqrt(self.reward_rms.var) + 1e-8)
        return reward * self.config['reward_scale']
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        
        rewards = rewards.cpu().numpy()
        values = values.cpu().numpy()
        dones = dones.cpu().numpy()
        next_value = next_value.cpu().item()
        
        values = np.append(values, next_value)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.config['gamma'] * next_value_t * next_non_terminal - values[t]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * next_non_terminal * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32).to(self.device)
        
        return advantages, returns
    
    def collect_rollout(self, n_episodes=1):
        self.buffer.clear()
        episode_infos = []
        
        for _ in range(n_episodes):
            ep_idx = np.random.randint(0, len(self.env.episodes))
            state = self.env.reset(episode_index=ep_idx)
            done = False
            
            ep_reward = 0
            ep_reward_raw = 0
            ep_length = 0
            
            while not done:
                state_tensor = torch.FloatTensor(state).to(self.device)
                
                with torch.no_grad():
                    action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
                
                next_state, reward_raw, done, info = self.env.step(action.cpu().numpy())
                
                # Normalize reward
                reward = self.normalize_reward(reward_raw)
                
                self.buffer.add(
                    state=state_tensor,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=done
                )
                
                ep_reward += reward
                ep_reward_raw += reward_raw
                ep_length += 1
                
                if not done:
                    state = next_state
            
            episode_infos.append({
                'reward': ep_reward,
                'reward_raw': ep_reward_raw,
                'length': ep_length,
                'final_value': info['portfolio_value']
            })
        
        return episode_infos
    
    def update_policy(self):
        data = self.buffer.get()
        
        with torch.no_grad():
            last_state = data['states'][-1]
            _, next_value = self.network(last_state.unsqueeze(0))
            next_value = next_value.squeeze()
        
        advantages, returns = self.compute_gae(
            data['rewards'], data['values'], data['dones'], next_value
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(data['states'])
        indices = np.arange(dataset_size)
        
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []
        
        for epoch in range(self.config['n_epochs']):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.config['batch_size']):
                end = start + self.config['batch_size']
                batch_indices = indices[start:end]
                
                batch_states = data['states'][batch_indices]
                batch_actions = data['actions'][batch_indices]
                batch_old_log_probs = data['log_probs'][batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 
                                   1 + self.config['clip_epsilon']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Track clipping
                clip_fraction = ((ratio - 1.0).abs() > self.config['clip_epsilon']).float().mean()
                
                # Value loss (clipped for stability)
                value_pred_clipped = data['values'][batch_indices] + torch.clamp(
                    new_values - data['values'][batch_indices],
                    -self.config['clip_epsilon'],
                    self.config['clip_epsilon']
                )
                value_loss_unclipped = ((new_values - batch_returns) ** 2)
                value_loss_clipped = ((value_pred_clipped - batch_returns) ** 2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config['value_coef'] * value_loss - 
                       self.config['entropy_coef'] * entropy.mean())
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 
                                        self.config['max_grad_norm'])
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                clip_fractions.append(clip_fraction.item())
        
        # Learning rate scheduling
        self.scheduler.step()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate(self, n_episodes=None):
        if n_episodes is None:
            n_episodes = len(self.env.episodes)
        
        self.network.eval()
        eval_rewards = []
        eval_values = []
        eval_sharpes = []
        
        for ep_idx in range(min(n_episodes, len(self.env.episodes))):
            state = self.env.reset(episode_index=ep_idx)
            done = False
            ep_reward = 0
            ep_returns = []
            
            while not done:
                state_tensor = torch.FloatTensor(state).to(self.device)
                
                with torch.no_grad():
                    concentrations, _ = self.network(state_tensor)
                    # Deterministic: use mode (mean for Dirichlet)
                    action = concentrations / concentrations.sum()
                
                state, reward, done, info = self.env.step(action.cpu().numpy())
                ep_reward += reward
                ep_returns.append(reward)
            
            eval_rewards.append(ep_reward)
            eval_values.append(info['portfolio_value'])
            
            # Calculate Sharpe
            if len(ep_returns) > 1:
                sharpe = np.mean(ep_returns) / (np.std(ep_returns) + 1e-8) * np.sqrt(52)
                eval_sharpes.append(sharpe)
        
        self.network.train()
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_value': np.mean(eval_values),
            'std_value': np.std(eval_values),
            'mean_sharpe': np.mean(eval_sharpes) if eval_sharpes else 0.0,
        }
    
    def train(self, n_iterations=300, rollout_episodes=5):
        print("\n" + "="*70)
        print("Starting IMPROVED PPO Training")
        print("="*70)
        
        best_reward = -np.inf
        
        for iteration in range(n_iterations):
            # Collect rollouts
            episode_infos = self.collect_rollout(n_episodes=rollout_episodes)
            
            # Update policy
            update_info = self.update_policy()
            
            # Track metrics
            mean_reward = np.mean([ep['reward'] for ep in episode_infos])
            mean_reward_raw = np.mean([ep['reward_raw'] for ep in episode_infos])
            mean_length = np.mean([ep['length'] for ep in episode_infos])
            mean_value = np.mean([ep['final_value'] for ep in episode_infos])
            
            self.episode_rewards.append(mean_reward_raw)
            self.episode_lengths.append(mean_length)
            self.episode_values.append(mean_value)
            self.policy_losses.append(update_info['policy_loss'])
            self.value_losses.append(update_info['value_loss'])
            self.entropies.append(update_info['entropy'])
            
            # Logging
            if (iteration + 1) % 10 == 0:
                print(f"\nIteration {iteration + 1}/{n_iterations}")
                print(f"  Reward (scaled): {mean_reward:.4f} | Raw: {mean_reward_raw:.4f}")
                print(f"  Length: {mean_length:.1f} | Value: ₹{mean_value:,.0f}")
                print(f"  Policy loss: {update_info['policy_loss']:.4f} | Value loss: {update_info['value_loss']:.4f}")
                print(f"  Entropy: {update_info['entropy']:.4f} | Clip frac: {update_info['clip_fraction']:.3f}")
                print(f"  Learning rate: {update_info['lr']:.2e}")
            
            # Evaluation
            if (iteration + 1) % self.config['eval_frequency'] == 0:
                eval_result = self.evaluate()
                self.eval_results.append({
                    'iteration': iteration + 1,
                    **eval_result
                })
                print(f"  [EVAL] Reward: {eval_result['mean_reward']:.4f} ± {eval_result['std_reward']:.4f}")
                print(f"  [EVAL] Value: ₹{eval_result['mean_value']:,.0f} ± ₹{eval_result['std_value']:,.0f}")
                print(f"  [EVAL] Sharpe: {eval_result['mean_sharpe']:.2f}")
                
                if eval_result['mean_reward'] > best_reward:
                    best_reward = eval_result['mean_reward']
                    self.save_checkpoint('best_model.pt')
                    print(f"  ✅ New best model! (reward: {best_reward:.4f})")
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
    
    def save_checkpoint(self, filename='checkpoint.pt'):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = CHECKPOINT_DIR / filename
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'eval_results': self.eval_results,
            'reward_rms': self.reward_rms,
        }, path)
    
    def load_checkpoint(self, filename='checkpoint.pt'):
        path = CHECKPOINT_DIR / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.eval_results = checkpoint.get('eval_results', [])
        self.reward_rms = checkpoint.get('reward_rms', None)
        
        print(f"Checkpoint loaded from {path}")

# ---------- Visualization ----------
def plot_training_curves(trainer, save_path=None):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    
    # Rewards
    axes[0, 0].plot(trainer.episode_rewards, alpha=0.6, label='Training')
    if trainer.eval_results:
        eval_iters = [r['iteration'] for r in trainer.eval_results]
        eval_rewards = [r['mean_reward'] for r in trainer.eval_results]
        axes[0, 0].plot(eval_iters, eval_rewards, 'r-', linewidth=2, label='Evaluation')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].set_title('Training Rewards (Raw)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Portfolio values
    axes[0, 1].plot(trainer.episode_values, alpha=0.6, label='Training')
    if trainer.eval_results:
        eval_values = [r['mean_value'] for r in trainer.eval_results]
        axes[0, 1].plot(eval_iters, eval_values, 'r-', linewidth=2, label='Evaluation')
    axes[0, 1].axhline(y=INITIAL_CAPITAL, color='g', linestyle='--', alpha=0.5, label='Initial')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Portfolio Value (₹)')
    axes[0, 1].set_title('Portfolio Performance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Losses
    axes[1, 0].plot(trainer.policy_losses, label='Policy Loss', alpha=0.7)
    axes[1, 0].plot(trainer.value_losses, label='Value Loss', alpha=0.7)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 1].plot(trainer.entropies, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Policy Entropy (should be positive!)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Smoothed rewards
    if len(trainer.episode_rewards) > 20:
        window = 20
        smoothed = np.convolve(trainer.episode_rewards, np.ones(window)/window, mode='valid')
        axes[2, 0].plot(smoothed, linewidth=2, color='darkblue')
        axes[2, 0].set_xlabel('Iteration')
        axes[2, 0].set_ylabel('Smoothed Reward')
        axes[2, 0].set_title(f'Smoothed Rewards (window={window})')
        axes[2, 0].grid(True, alpha=0.3)
    
    # Sharpe ratios
    if trainer.eval_results:
        eval_sharpes = [r['mean_sharpe'] for r in trainer.eval_results]
        axes[2, 1].plot(eval_iters, eval_sharpes, 'g-', linewidth=2, marker='o')
        axes[2, 1].set_xlabel('Iteration')
        axes[2, 1].set_ylabel('Sharpe Ratio')
        axes[2, 1].set_title('Evaluation Sharpe Ratios')
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
    
    plt.show()

# ---------- MAIN ----------
if __name__ == "__main__":
    print("Loading dataset...")
    dataset_npz = np.load(RL_DATASET_PATH, allow_pickle=True)
    dataset = {
        "states": dataset_npz["states"],
        "prices": dataset_npz["prices"],
        "rewards": dataset_npz["rewards"],
        "episodes": dataset_npz["episodes"].tolist(),
        "tickers": dataset_npz["tickers"].tolist(),
        "macro_tickers": dataset_npz["macro_tickers"].tolist(),
        "features_trad": dataset_npz["features_trad"].tolist(),
        "features_macro": dataset_npz["features_macro"].tolist(),
    }
    
    print(f"Dataset: {len(dataset['episodes'])} episodes, {len(dataset['tickers'])} assets")
    
    env = PortfolioEnv(dataset, initial_capital=INITIAL_CAPITAL)
    trainer = PPOTrainer(env, config=PPO_CONFIG)
    
    trainer.train(n_iterations=PPO_CONFIG['n_training_episodes'], rollout_episodes=5)
    
    trainer.save_checkpoint('final_model.pt')
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'config': PPO_CONFIG,
        'episode_rewards': trainer.episode_rewards,
        'episode_values': trainer.episode_values,
        'policy_losses': trainer.policy_losses,
        'value_losses': trainer.value_losses,
        'entropies': trainer.entropies,
        'eval_results': trainer.eval_results,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(RESULTS_DIR / 'training_history.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_training_curves(trainer, save_path=RESULTS_DIR / 'training_curves.png')
    
    print("\n" + "="*70)
    print("Training complete! Files saved:")
    print(f"  - Best model: {CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"  - Final model: {CHECKPOINT_DIR / 'final_model.pt'}")
    print(f"  - Training history: {RESULTS_DIR / 'training_history.json'}")
    print(f"  - Training curves: {RESULTS_DIR / 'training_curves.png'}")
    print("="*70)