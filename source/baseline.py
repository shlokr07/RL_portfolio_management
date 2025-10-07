# source/baseline.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rl_data_prep import PortfolioEnv

# ---------- CONFIG ----------
RL_DATASET_PATH = Path("data/processed/rl_dataset.npz")
INITIAL_CAPITAL = 1_000_000  # Starting capital for baseline (should match env default)
RESULTS_PATH = Path("data/processed/baseline_results.npz")

# ---------- Utility Functions ----------
def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=52):
    """Calculate annualized Sharpe ratio from weekly returns"""
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    if np.std(excess_returns) == 0:
        return 0.0
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown from portfolio value series"""
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return np.min(drawdown)

def calculate_metrics(portfolio_values):
    """Calculate comprehensive performance metrics"""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0
    annualized_return = (1 + total_return) ** (52 / len(portfolio_values)) - 1.0
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(portfolio_values)
    volatility = np.std(returns) * np.sqrt(52)  # Annualized
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "volatility": volatility,
        "final_value": portfolio_values[-1],
    }

# ---------- Load Dataset ----------
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

print(f"Dataset loaded: {len(dataset['episodes'])} episodes, {len(dataset['tickers'])} assets")

# ---------- Initialize Environment ----------
env = PortfolioEnv(dataset, initial_capital=INITIAL_CAPITAL)

# ---------- Equal Allocation Baseline ----------
print("\n" + "="*60)
print("Running Equal Weight Baseline Strategy")
print("="*60)

all_episode_results = []

for ep_idx, (start, end) in enumerate(dataset["episodes"]):
    obs = env.reset(episode_index=ep_idx)
    done = False
    step = 0
    
    portfolio_values = [env.initial_capital]  # Start with initial capital
    weekly_returns = []

    # Uniform allocation: equal weight to all assets
    action = np.ones(env.n_assets) / env.n_assets

    while not done:
        obs, reward, done, info = env.step(action)
        
        # Track portfolio value and returns
        portfolio_values.append(info["portfolio_value"])
        weekly_returns.append(reward)
        step += 1
        
        # Safety check to prevent infinite loops
        if step > (end - start) + 10:
            print(f"  Warning: Episode {ep_idx} exceeded expected length, breaking")
            break

    # Calculate metrics for this episode
    portfolio_values = np.array(portfolio_values)
    weekly_returns = np.array(weekly_returns)
    metrics = calculate_metrics(portfolio_values)
    
    episode_result = {
        "episode_index": ep_idx,
        "start": start,
        "end": end,
        "n_steps": len(portfolio_values) - 1,
        "portfolio_values": portfolio_values,
        "weekly_returns": weekly_returns,
        **metrics
    }
    
    all_episode_results.append(episode_result)
    
    print(f"\nEpisode {ep_idx} ({start} to {end}):")
    print(f"  Steps: {len(portfolio_values)-1}")
    print(f"  Final value: ₹{metrics['final_value']:,.2f}")
    print(f"  Total return: {metrics['total_return']*100:.2f}%")
    print(f"  Ann. return: {metrics['annualized_return']*100:.2f}%")
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Volatility: {metrics['volatility']*100:.2f}%")

# ---------- Aggregate Statistics ----------
print("\n" + "="*60)
print("Aggregate Performance Across All Episodes")
print("="*60)

total_returns = [ep["total_return"] for ep in all_episode_results]
sharpe_ratios = [ep["sharpe_ratio"] for ep in all_episode_results]
max_drawdowns = [ep["max_drawdown"] for ep in all_episode_results]

print(f"Mean total return: {np.mean(total_returns)*100:.2f}% (±{np.std(total_returns)*100:.2f}%)")
print(f"Mean Sharpe ratio: {np.mean(sharpe_ratios):.3f} (±{np.std(sharpe_ratios):.3f})")
print(f"Mean max drawdown: {np.mean(max_drawdowns)*100:.2f}% (±{np.std(max_drawdowns)*100:.2f}%)")

# ---------- Save Results ----------
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Prepare data for saving (numpy-compatible format)
save_dict = {
    "n_episodes": len(all_episode_results),
    "n_assets": env.n_assets,
    "initial_capital": INITIAL_CAPITAL,
    "aggregate_mean_return": np.mean(total_returns),
    "aggregate_mean_sharpe": np.mean(sharpe_ratios),
    "aggregate_mean_drawdown": np.mean(max_drawdowns),
}

# Save each episode's data separately
for i, ep_res in enumerate(all_episode_results):
    save_dict[f"ep{i}_portfolio_values"] = ep_res["portfolio_values"]
    save_dict[f"ep{i}_returns"] = ep_res["weekly_returns"]
    save_dict[f"ep{i}_total_return"] = ep_res["total_return"]
    save_dict[f"ep{i}_sharpe"] = ep_res["sharpe_ratio"]
    save_dict[f"ep{i}_max_dd"] = ep_res["max_drawdown"]

np.savez_compressed(RESULTS_PATH, **save_dict)
print(f"\n✅ Baseline results saved to {RESULTS_PATH}")

# ---------- Plot Portfolio Value Curves ----------
n_episodes = len(all_episode_results)
n_cols = min(3, n_episodes)
n_rows = (n_episodes + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
if n_episodes == 1:
    axes = np.array([axes])
axes = axes.flatten()

for idx, ep_res in enumerate(all_episode_results):
    ax = axes[idx]
    weeks = np.arange(len(ep_res["portfolio_values"]))
    ax.plot(weeks, ep_res["portfolio_values"], linewidth=2)
    ax.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
    ax.set_xlabel("Week")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.set_title(f"Episode {idx}\nReturn: {ep_res['total_return']*100:.1f}%, Sharpe: {ep_res['sharpe_ratio']:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend()

# Hide unused subplots
for idx in range(n_episodes, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(RESULTS_PATH.parent / "baseline_portfolio_curves.png", dpi=150)
print(f"✅ Portfolio curves saved to {RESULTS_PATH.parent / 'baseline_portfolio_curves.png'}")
plt.show()

# ---------- Plot Returns Distribution ----------
plt.figure(figsize=(10, 6))
all_returns = np.concatenate([ep["weekly_returns"] for ep in all_episode_results])
plt.hist(all_returns, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Return')
plt.axvline(x=np.mean(all_returns), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_returns)*100:.2f}%')
plt.xlabel("Weekly Return")
plt.ylabel("Frequency")
plt.title("Distribution of Weekly Returns (Equal Weight Strategy)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_PATH.parent / "baseline_returns_distribution.png", dpi=150)
print(f"✅ Returns distribution saved to {RESULTS_PATH.parent / 'baseline_returns_distribution.png'}")
plt.show()

print("\n" + "="*60)
print("Baseline evaluation complete!")
print("="*60)