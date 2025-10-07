# src/rl_data_env.py
import numpy as np
import pandas as pd
from pathlib import Path
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# ---------- CONFIG ----------
HDF_PATH = Path("data/processed/features_weekly.h5")
RL_DATASET_PATH = Path("data/processed/rl_dataset.npz")
TICKERS_GROUP = "tradables"
MACROS_GROUP = "macros"

WEEKS_PER_EPISODE = 52
INITIAL_CAPITAL = 100_000  # 1 lakh starting capital

# ---------- RL Dataset Preparation ----------
class RLDataPreparer:
    def __init__(self, hdf_path=HDF_PATH):
        self.hdf_path = hdf_path
        self.tradables = {}
        self.macros = {}
    
    def read_hdf_features(self):
        """Load tradables and macro features into dicts of DataFrames"""
        with pd.HDFStore(self.hdf_path, mode="r") as store:
            for key in store.keys():
                clean_key = key.strip("/")
                if clean_key.startswith(TICKERS_GROUP):
                    ticker = clean_key.split("/")[-1]
                    self.tradables[ticker] = store[key]
                elif clean_key.startswith(MACROS_GROUP):
                    ticker = clean_key.split("/")[-1]
                    self.macros[ticker] = store[key]
        return self.tradables, self.macros
    
    @staticmethod
    def build_3d_array(feature_dict):
        tickers = sorted(feature_dict.keys())
        dfs = [feature_dict[t] for t in tickers]

        base_index = dfs[0].index
        for df in dfs[1:]:
            assert (df.index == base_index).all(), "Indices are not aligned!"
        
        array_3d = np.stack([df.values for df in dfs], axis=1)
        return array_3d, tickers, list(dfs[0].columns), base_index

    @staticmethod
    def compute_weekly_returns(prices_array):
        """Compute percentage returns week-over-week"""
        returns = (prices_array[1:] - prices_array[:-1]) / (prices_array[:-1] + 1e-8)
        return returns

    def create_rl_dataset(self, weeks_per_episode=WEEKS_PER_EPISODE):
        X_trad, tickers_trad, features_trad, idx_trad = self.build_3d_array(self.tradables)
        X_macro, tickers_macro, features_macro, idx_macro = self.build_3d_array(self.macros)

        assert (idx_trad == idx_macro).all(), "Tradable and macro dates are not aligned!"

        # Extract prices (first column) separately
        prices = X_trad[:, :, 0]  # Shape: (T, N_assets)
        
        # Remove raw close from features, keep only normalized features
        X_trad_features = X_trad[:, :, 1:]  # Skip first column (raw close)
        features_trad_used = features_trad[1:]  # Update feature names
        
        # Flatten macro features and broadcast to all assets
        X_macro_flat = X_macro.reshape(X_macro.shape[0], -1)
        X_macro_exp = np.repeat(X_macro_flat[:, np.newaxis, :], X_trad_features.shape[1], axis=1)
        
        # Concatenate tradable features (without raw close) with macro features
        X = np.concatenate([X_trad_features, X_macro_exp], axis=2)

        # Compute returns for rewards
        rewards = self.compute_weekly_returns(prices)
        
        # Align: we need prices[t] to compute position sizes, 
        # features[t] to make decisions, and rewards[t] = return from t to t+1
        # Drop last timestep since we don't have future returns for it
        X = X[:-1, :, :]
        prices = prices[:-1, :]
        
        # Verify shapes
        assert X.shape[0] == rewards.shape[0], f"State/reward mismatch: {X.shape[0]} vs {rewards.shape[0]}"
        assert prices.shape[0] == rewards.shape[0], f"Price/reward mismatch: {prices.shape[0]} vs {rewards.shape[0]}"

        # Create episodes (non-overlapping windows)
        total_weeks = X.shape[0]
        episodes = [
            (start, min(start + weeks_per_episode, total_weeks))
            for start in range(0, total_weeks, weeks_per_episode)
        ]
        # Remove last episode if it's too short (< 10 weeks)
        if episodes and (episodes[-1][1] - episodes[-1][0]) < 10:
            episodes = episodes[:-1]

        dataset = {
            "states": X,
            "prices": prices,
            "rewards": rewards,
            "episodes": episodes,
            "tickers": tickers_trad,
            "macro_tickers": tickers_macro,
            "features_trad": features_trad_used,
            "features_macro": features_macro,
            "dates": idx_trad[:-1].tolist(),  # Align with states
        }

        print(f"Dataset shapes:")
        print(f"  States: {X.shape} (time, assets, features)")
        print(f"  Prices: {prices.shape} (time, assets)")
        print(f"  Rewards: {rewards.shape} (time, assets)")
        print(f"  Episodes: {len(episodes)} episodes")
        print(f"  Features per asset: {X.shape[2]} ({len(features_trad_used)} tradable + {X_macro_flat.shape[1]} macro)")

        return dataset

    @staticmethod
    def save_dataset(dataset, path=RL_DATASET_PATH):
        np.savez_compressed(
            path,
            states=dataset["states"],
            prices=dataset["prices"],
            rewards=dataset["rewards"],
            episodes=np.array(dataset["episodes"]),
            tickers=np.array(dataset["tickers"]),
            macro_tickers=np.array(dataset["macro_tickers"]),
            features_trad=np.array(dataset["features_trad"]),
            features_macro=np.array(dataset["features_macro"]),
        )
        print(f"✅ RL dataset saved at {path}")


# ---------- Gym-Compatible Environment ----------
class PortfolioEnv(gym.Env):
    """
    Portfolio allocation environment with capital constraints.
    
    State: Normalized features for each asset (no raw prices)
    Action: Target portfolio weights (will be converted to shares based on prices & capital)
    Reward: Portfolio return for the week
    """
    
    def __init__(self, rl_dataset, initial_capital=INITIAL_CAPITAL):
        super().__init__()
        self.states = rl_dataset["states"]
        self.prices = rl_dataset["prices"]
        self.rewards = rl_dataset["rewards"]
        self.episodes = rl_dataset["episodes"]
        self.n_assets = self.states.shape[1]
        self.n_features = self.states.shape[2]
        self.initial_capital = initial_capital

        # Action space: target portfolio weights (continuous, will be normalized)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Observation space: (N_assets, F_features) - NO raw prices
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_assets, self.n_features), dtype=np.float32
        )

        self.current_episode = 0
        self.current_step = 0
        self.cash = initial_capital
        self.portfolio_value = initial_capital

    def reset(self, episode_index=0):
        """Reset environment to start of episode"""
        if episode_index >= len(self.episodes):
            episode_index = 0
        
        self.current_episode = episode_index
        self.current_step = self.episodes[self.current_episode][0]
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        
        obs = self.states[self.current_step].astype(np.float32)
        return obs

    def step(self, action):
        """
        Execute one time step.
        
        Action represents target portfolio weights.
        We convert weights to shares based on current prices and available capital.
        """
        if self.current_step >= self.episodes[self.current_episode][1] - 1:
            # Episode already done
            return self.states[self.current_step].astype(np.float32), 0.0, True, {}
        
        # Normalize action to valid portfolio weights
        action = np.clip(action, 0.0, 1.0)
        action = action / (np.sum(action) + 1e-8)
        
        # Get current prices for position sizing
        current_prices = self.prices[self.current_step]
        
        # Convert weights to capital allocation (respecting capital constraint)
        target_capital_per_asset = action * self.portfolio_value
        
        # Compute shares (integer shares for realism, or keep float for simplicity)
        shares = target_capital_per_asset / (current_prices + 1e-8)
        
        # Actual capital allocated (in case of rounding)
        actual_capital_per_asset = shares * current_prices
        total_invested = np.sum(actual_capital_per_asset)
        
        # Update cash (leftover after allocation)
        self.cash = self.portfolio_value - total_invested
        
        # Compute portfolio return: weighted average of asset returns
        asset_returns = self.rewards[self.current_step]
        portfolio_return = np.dot(asset_returns, action)
        
        # Update portfolio value
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.episodes[self.current_episode][1] - 1
        
        obs = None if done else self.states[self.current_step].astype(np.float32)
        
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "total_invested": total_invested,
            "shares": shares,
        }
        
        return obs, portfolio_return, done, info

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Portfolio Value: ₹{self.portfolio_value:,.2f}, Cash: ₹{self.cash:,.2f}")


# ---------- MAIN ----------
if __name__ == "__main__":
    preparer = RLDataPreparer()
    preparer.read_hdf_features()
    dataset = preparer.create_rl_dataset(weeks_per_episode=WEEKS_PER_EPISODE)
    preparer.save_dataset(dataset)

    # Example environment usage
    env = PortfolioEnv(dataset, initial_capital=INITIAL_CAPITAL)
    obs = env.reset()
    print("\n" + "="*60)
    print("Environment Test:")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Features per asset: {obs.shape[1]}")
    print(f"Number of assets: {env.n_assets}")
    print(f"Initial capital: ₹{env.initial_capital:,}")
    
    # Test a uniform allocation
    action = np.ones(env.n_assets) / env.n_assets
    obs, reward, done, info = env.step(action)
    print(f"\nAfter first step:")
    print(f"  Reward (portfolio return): {reward:.4%}")
    print(f"  Portfolio value: ₹{info['portfolio_value']:,.2f}")
    print(f"  Cash remaining: ₹{info['cash']:,.2f}")
    print(f"  Total invested: ₹{info['total_invested']:,.2f}")