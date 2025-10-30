import gym
import numpy as np

class EnergyEnv(gym.Env):
    def __init__(self):
        super(EnergyEnv, self).__init__()
        # Define action and state space
        self.action_space = gym.spaces.Discrete(3)  # Example: 0 - discharge battery, 1 - charge battery, 2 - use grid
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)  # [battery SoC, solar output, wind output, demand, time]

        # Initialize state variables
        self.battery_soc = 0.5  # Battery state of charge (50% initially)
        self.solar_output = 0.0  # Solar output
        self.wind_output = 0.0  # Wind output
        self.demand = 1.0  # Demand
        self.time = 0  # Time step

    def reset(self):
        self.battery_soc = 0.5
        self.solar_output = np.random.uniform(0, 1)  # Random solar output
        self.wind_output = np.random.uniform(0, 1)  # Random wind output
        self.demand = np.random.uniform(0.5, 1.5)  # Random demand
        self.time = 0
        return np.array([self.battery_soc, self.solar_output, self.wind_output, self.demand, self.time], dtype=np.float32)

    def step(self, action):
        # Implement action logic
        if action == 0:  # Discharge battery
            self.battery_soc -= 0.1
        elif action == 1:  # Charge battery
            self.battery_soc += 0.1
        elif action == 2:  # Use grid
            pass

        # Update state
        self.battery_soc = np.clip(self.battery_soc, 0, 1)  # Keep SOC between 0 and 1
        self.time += 1

        # Calculate reward
        cost = self.demand - (self.solar_output + self.wind_output + (self.battery_soc * 100))  # Simplified cost calculation
        reward = -cost  # Reward is negative cost (minimization)

        done = self.time >= 24  # End after 24 time steps
        return np.array([self.battery_soc, self.solar_output, self.wind_output, self.demand, self.time], dtype=np.float32), reward, done, {}

    def render(self, mode='human'):  # Optional visualization
        pass  # Implement visualization if needed
