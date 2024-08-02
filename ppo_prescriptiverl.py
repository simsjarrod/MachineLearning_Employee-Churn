import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO  # Proximal Policy Optimization algorithm
from stable_baselines3.common.env_checker import check_env

class CSVEnv(gym.Env):
    def __init__(self, csv):
        super(CSVEnv, self).__init__()

        # store data loaded from csv
        self.data = pd.read_csv(csv)
        
        # Define actions and observation space
        num_features = self.data.shape[1] - 1
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.state = self.data.iloc[self.current_step].values[:-1].astype(np.float32)  # Exclude satisfaction from state
        return self.state, {}

    def step(self, action):
        if action == 0:
            self.data.loc[self.current_step, 'MonthlyIncome'] *= 1.1  # Increase salary by 10%
        elif action == 1:
            self.data.loc[self.current_step, 'StockOptionLevel'] += 1  # Increase employee stock options
        elif action == 2:
            self.data.loc[self.current_step, 'OverTime'] *= 0  # Eliminate overtime

        reward = self.calculate_reward(action)
        
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
            truncated = True
            self.current_step -= 1  # Ensure last state is maintained for rendering
        else:
            done = False
            truncated = False
            self.state = self.data.iloc[self.current_step].values[:-1].astype(np.float32)  # Exclude satisfaction from state
            
        return self.state, reward, done, truncated, {}

    def calculate_reward(self, action):
        satisfaction_before = self.data.loc[self.current_step, 'JobSatisfaction']
        
        # Hypothetical model for satisfaction change
        if action == 0:  # Increase salary
            satisfaction_after = satisfaction_before + 0.05
        elif action == 1:  # Increase stock options
            satisfaction_after = satisfaction_before + 0.1
        elif action == 2:  # Reduce overtime
            satisfaction_after = satisfaction_before + 0.2
        
        reward = satisfaction_after - satisfaction_before
        self.data.loc[self.current_step, 'JobSatisfaction'] = satisfaction_after
        
        return reward

    def render(self, mode='human'):
        action_names = ["Increase Salary", "Increase Stock Options", "Promote Individual"]
        print(f"Step: {self.current_step}, State: {self.state}, Action: {action_names[action]}")

    def close(self):
        pass

# Load and preprocess high risk employee csv
df = pd.read_csv('high_risk_employees.csv')
df.to_csv('ppo_data.csv', index=False)

# Use the custom environment
env = CSVEnv('ppo_data.csv')
check_env(env)  # Check the environment to make sure it is valid

# Train the RL agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# List to store action recommendations
recommendations = []

# Use the trained agent to find the best action for each employee
for i in range(len(df)):
    state = env.reset()[0]  # Ensure to use the state only
    action, _ = model.predict(state)
    action_names = ["Increase Salary 10%", "Increase Stock Options 1 level", "Eliminate Overtime"]
    action_name = action_names[action]
    recommendations.append({'Employee': i+1, 'Action': action_name})

# Convert recommendations to DataFrame and export to CSV
recommendations_df = pd.DataFrame(recommendations)
recommendations_df.to_csv('high_risk_employee_action_recs.csv', index=False)

env.close()