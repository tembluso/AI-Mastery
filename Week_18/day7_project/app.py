# app.py
# Week 18 · Day 7 — LunarLander A2C mini-app (Train + Frame Slider with touchdown capture)
# ────────────────────────────────────────────────────────────────────────────────
# How to run (Windows/macOS/Linux):
#   1) (Recommended) Create/activate a virtual env
#   2) pip install streamlit torch numpy matplotlib gymnasium pygame imageio ufal.pybox2d
#   3) streamlit run app.py
#
# Notes:
# - Uses Gymnasium's LunarLander (Box2D). We render with render_mode="rgb_array".
# - We COPY each frame buffer so the slider shows the real timeline (touchdown/crash).
# - We also capture a final frame after termination, and offer "best-of-N" evaluation to
#   more reliably show a full landing.

import io, random
from collections import deque
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ─── Global setup ───────────────────────────────────────────────────────────────
np.random.seed(42); random.seed(42); torch.manual_seed(42)
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["axes.grid"] = True

st.set_page_config(page_title="LunarLander — Train & Replay", layout="wide")
st.title("🚀 LunarLander — Train (A2C) & Replay with Frame Slider")

# ─── Sidebar configuration ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Training config")
    episodes = st.slider("Episodes", 50, 5000, 800, step=50)
    lr = st.select_slider("Learning Rate", options=[5e-4, 1e-3, 2e-3], value=1e-3)
    hidden = st.select_slider("Hidden Size", options=[128, 256, 512], value=256)
    gamma = st.slider("Discount γ", 0.90, 0.999, 0.99)
    critic_coef = st.select_slider("Critic Coef", options=[0.2, 0.5, 1.0], value=0.5)
    entropy_beta = st.select_slider("Entropy β", options=[0.0, 1e-3, 5e-3, 1e-2], value=1e-3)
    seed = st.number_input("Seed", value=42, step=1)

    st.divider()
    st.header("Evaluation config")
    eval_episodes = st.slider("Best-of-N evaluation episodes", 1, 10, 3)
    max_steps = st.slider("Max steps per eval", 200, 3000, 1500, step=100)
    downsample_to = st.slider("Downsample frames to (for slider performance)", 100, 600, 300, step=50)

    st.divider()
    train_btn = st.button("🚀 Train A2C")
    eval_btn = st.button("🎬 Evaluate & Capture (best of N)")
    reset_btn = st.button("♻️ Reset session state")

# ─── Utilities ─────────────────────────────────────────────────────────────────
ENV_ID = "LunarLander-v3"  # use v3 for widest compatibility; v3 works if available in your Gymnasium

def make_env(render=False, seed_=42):
    render_mode = "rgb_array" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode)
    try:
        env.reset(seed=seed_)
    except TypeError:
        pass
    return env

def moving_average(x, w=50):
    if len(x) < w: return np.array([])
    return np.convolve(x, np.ones(w)/w, mode="valid")

def downsample_frames(frames, target=300):
    if len(frames) <= target:
        return frames
    step = max(1, len(frames) // target)
    return frames[::step]

# ─── A2C Model ─────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(hidden, n_actions), nn.Softmax(dim=-1))
        self.value = nn.Linear(hidden, 1)
    def forward(self, x):
        z = self.shared(x)
        return self.policy(z), self.value(z)

class A2CAgent:
    def __init__(self, obs_dim, n_actions, hidden=256, lr=1e-3, gamma=0.99, critic_coef=0.5, entropy_beta=1e-3):
        self.model = ActorCritic(obs_dim, n_actions, hidden=hidden)
        self.optim = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.critic_coef = critic_coef
        self.entropy_beta = entropy_beta

    @torch.no_grad()
    def act_greedy(self, obs):
        s_t = torch.tensor(obs, dtype=torch.float32)
        probs, _ = self.model(s_t)
        return int(probs.argmax().item())

    def run_episode(self, env):
        logps, vals, rews, ents = [], [], [], []
        s, _ = env.reset()
        done = False; ep_ret = 0.0
        while not done:
            s_t = torch.tensor(s, dtype=torch.float32)
            probs, value = self.model(s_t)
            m = Categorical(probs)
            a = m.sample()
            s, r, term, trunc, _ = env.step(int(a.item()))
            done = term or trunc

            logps.append(m.log_prob(a))
            vals.append(value)
            rews.append(r)
            ents.append(m.entropy())
            ep_ret += r
        return logps, vals, rews, ents, ep_ret

    def compute_returns(self, rewards):
        G, out = 0.0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            out.append(G)
        out.reverse()
        ret = torch.tensor(out, dtype=torch.float32)
        if ret.std() > 0:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        return ret

    def train_episode(self, env):
        logps, vals, rews, ents, ep_ret = self.run_episode(env)
        returns = self.compute_returns(rews)
        vals = torch.cat(vals).squeeze()
        adv = returns - vals.detach()

        actor_loss = -(torch.stack(logps) * adv).mean()
        critic_loss = (returns - vals).pow(2).mean()
        entropy_loss = -torch.stack(ents).mean()

        loss = actor_loss + self.critic_coef * critic_loss + self.entropy_beta * entropy_loss
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return ep_ret

# ─── Training & Evaluation (with touchdown capture) ────────────────────────────
def train_agent(episodes, lr, hidden, gamma, critic_coef, entropy_beta, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = make_env(render=False, seed_=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = A2CAgent(obs_dim, n_actions, hidden=hidden, lr=lr, gamma=gamma,
                     critic_coef=critic_coef, entropy_beta=entropy_beta)

    rewards = []
    for ep in range(1, episodes + 1):
        ep_ret = agent.train_episode(env)
        rewards.append(ep_ret)
        if ep % max(episodes // 16, 25) == 0:
            st.sidebar.write(f"Ep {ep} • avg(last 50): {np.mean(rewards[-50:]):.1f}")
    env.close()
    return agent, rewards

def evaluate_and_capture(agent, max_steps=2000, seed=42):
    """Run ONE episode, copy each frame buffer, capture BEFORE each step and one FINAL frame."""
    env = make_env(render=True, seed_=seed)
    frames = []
    obs, _ = env.reset()
    total = 0.0
    for t in range(max_steps):
        # Capture current state frame (copy buffer!)
        frame = env.render()
        frames.append(np.asarray(frame, dtype=np.uint8).copy())

        # Greedy action at eval time
        action = agent.act_greedy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward

        if terminated or truncated:
            # Capture a final frame to show touchdown/crash state
            final = env.render()
            frames.append(np.asarray(final, dtype=np.uint8).copy())
            break
    env.close()
    return frames, total

def best_of_n(agent, n=3, max_steps=2000, seed=42):
    """Run N eval episodes with different seeds; return the one with the best total reward."""
    best_frames, best_total = None, -1e9
    for i in range(n):
        frames, total = evaluate_and_capture(agent, max_steps=max_steps, seed=seed + i)
        if total > best_total:
            best_total, best_frames = total, frames
    return best_frames, best_total

# ─── Session state ─────────────────────────────────────────────────────────────
if reset_btn:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.session_state.setdefault("trained", False)
st.session_state.setdefault("agent", None)
st.session_state.setdefault("rewards", [])
st.session_state.setdefault("frames", None)
st.session_state.setdefault("episode_total", None)

# ─── Train button ──────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Training A2C on LunarLander…"):
        agent, rewards = train_agent(
            episodes=episodes, lr=float(lr), hidden=hidden, gamma=gamma,
            critic_coef=critic_coef, entropy_beta=entropy_beta, seed=seed
        )
    st.session_state.agent = agent
    st.session_state.rewards = rewards
    st.session_state.trained = True
    st.success("Training complete ✅")

# ─── Plots (learning curve) ───────────────────────────────────────────────────
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Learning Curve")
    r = st.session_state.rewards
    if st.session_state.trained and len(r) > 0:
        fig, ax = plt.subplots()
        ax.plot(r, alpha=0.35, label="Episode reward")
        ma = moving_average(r, w=min(50, max(5, len(r)//10)))
        if len(ma) > 0:
            ax.plot(np.arange(len(r)-len(ma)+1, len(r)+1), ma, label="Moving Avg")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.legend()
        st.pyplot(fig)
    else:
        st.info("Train the agent to see the curve.")

with c2:
    st.subheader("Stats")
    if st.session_state.trained and len(st.session_state.rewards) > 0:
        arr = np.array(st.session_state.rewards)
        st.metric("Best Reward (single ep)", f"{arr.max():.1f}")
        tail = arr[-50:] if len(arr) >= 50 else arr
        st.metric("Mean Reward (last 50)" if len(arr) >= 50 else "Mean Reward", f"{tail.mean():.1f}")
        st.write(f"Episodes: **{len(arr)}** | Env: **{ENV_ID}** | Algo: **A2C**")
    else:
        st.info("No stats yet.")

st.divider()
st.subheader("🎬 Replay — Frame Slider (captures touchdown/crash)")

# ─── Evaluate button ──────────────────────────────────────────────────────────
if eval_btn:
    if not st.session_state.trained or st.session_state.agent is None:
        st.warning("Please train the agent first.")
    else:
        with st.spinner("Evaluating best-of-N and capturing frames…"):
            frames, total = best_of_n(st.session_state.agent, n=eval_episodes, max_steps=max_steps, seed=seed)
        # Downsample for snappy slider UX
        frames = [np.asarray(f, dtype=np.uint8) for f in frames]
        frames = downsample_frames(frames, target=downsample_to)
        st.session_state.frames = frames
        st.session_state.episode_total = total
        st.success(f"Captured {len(frames)} frames. Episode total reward (best-of-{eval_episodes}): **{total:.1f}**")

# ─── Frame slider UI ──────────────────────────────────────────────────────────
if st.session_state.frames:
    frames = st.session_state.frames
    max_idx = len(frames) - 1
    col1, col2 = st.columns([2, 1], vertical_alignment="top")
    with col1:
        idx = st.slider("Scrub frames", 0, max_idx, 0)
        # Streamlit <=1.38 uses 'use_column_width', not 'use_container_width'
        st.image(frames[idx], caption=f"Frame {idx}/{max_idx}", use_column_width=True)
    with col2:
        st.write("**Episode Summary**")
        st.write(f"Frames: **{len(frames)}**")
        st.write(f"Total Reward: **{st.session_state.episode_total:.1f}**")
        st.caption("Tip: Increase 'Best-of-N' in the sidebar to better catch a full landing.")
else:
    st.info("Click **Evaluate & Capture** to record an episode and use the slider.")
