import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
from cartpole_wrapper import CartPoleWrapper

def record_video(env, model, video_folder="videos"):
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
    obs, _ = env.reset()
    while True:
      action, _states = model.predict(obs)
      obs, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      if done:
          break
    env.close()

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = CartPoleWrapper(env)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=200000)

record_video(env, model)
