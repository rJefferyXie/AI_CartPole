import gymnasium as gym
from gymnasium import Wrapper

class CartPoleWrapper(Wrapper):
  def __init__(self, env: gym.Env):
    super().__init__(env)
    self.num_steps = 0
  
  def reset(self, **kwargs):
    self.num_steps = 0
    return self.env.reset(**kwargs)

  def step(self, action):
    obs, _, terminated, truncated, info = self.env.step(action)
    
    self.num_steps += 1

    reward = self.reward_function(obs, terminated, truncated)
    return obs, reward, terminated, truncated, info

  def reward_function(self, obs, terminated, truncated):
    reward = self.num_steps * 0.01 # rewarding the AI for total time spent

    cart_position = obs[0]
    cart_velocity = obs[1]
    pole_angle = obs[2]
    pole_angle_velocity = obs[3]

    reward += 1 - abs(pole_angle) - 0.1 * abs(pole_angle_velocity) # trying to keep pole vertical
    reward += 1 - abs(cart_position) # trying to stay at center of screen

    termination_penalty = -10 if terminated or truncated else 0

    print(f"Step: {self.num_steps:>4} | Reward: {reward:>7.3f} | Pole Angle: {pole_angle:>7.3f} | Cart Position: {cart_position:>7.3f}")
    return reward + termination_penalty