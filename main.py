from MedicalDecisionHeartFailureModel import reset, transition, states, rewards, actions
from MedicalDecisionHeartFailurePolicies import rule_based_policy, evaluate_policy
from PredictionModel import HeartFailurePrediction
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from sklearn.metrics import accuracy_score

# Neural Network for DQN
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Deep Q-Learning Agent
class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNetwork(state_size, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state_idx = np.argmax(state)
        valid_actions = []

        if states[state_idx] == "no_risk":
            valid_actions = [actions.index("no_treatment")]
        else:
            valid_actions = [actions.index("Aspirin"), actions.index("Warfarin"), actions.index("Rosuvastatin")]

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy().flatten()

            q_values = np.array([q_values[a] if a in valid_actions else -np.inf for a in range(len(actions))])
            return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.tensor(reward).float().to(self.device)

            target = self.model(state).detach()
            target_values = target.clone()

            if states[state.argmax()] == "no_risk":
                valid_actions = [actions.index("no_treatment")]
            else:
                valid_actions = [actions.index("Aspirin"), actions.index("Warfarin"), actions.index("Rosuvastatin")]

            for a in range(self.action_size):
                if a not in valid_actions:
                    target_values[0, a] = 0

            if done:
                target_values[0, action] = reward
            else:
                next_q_values = self.target_model(next_state).detach()
                target_values[0, action] = reward + self.gamma * torch.max(next_q_values).item()

            output = self.model(state)

            loss = self.criterion(output, target_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Function to generate Q-table
def generate_q_table(agent):
    q_table = []
    for state_idx in range(len(states)):
        state = np.eye(len(states))[state_idx].reshape(1, -1)
        state_tensor = torch.FloatTensor(state).to(agent.device)
        with torch.no_grad():
            q_values = agent.model(state_tensor).cpu().numpy().flatten()

        if states[state_idx] == "no_risk":
            q_values[[actions.index("Aspirin"), actions.index("Warfarin"), actions.index("Rosuvastatin")]] = 0
        else:
            q_values[actions.index("no_treatment")] = 0

        q_table.append(q_values)

    q_table_df = pd.DataFrame(q_table, columns=actions, index=states)
    return q_table_df

# Initialize Agent
state_size = len(states)
action_size = len(actions)
agent = DQLAgent(state_size, action_size)

# Training Parameters
episodes = 1000
batch_size = 32
reward_history = []
cumulative_reward_history = []
cumulative_reward = 0

# Rule-Based Policy Evaluation
rule_based_rewards = []
for e in range(episodes):
    rule_based_reward = evaluate_policy(rule_based_policy, rewards, states, transition, reset, episodes=1)
    rule_based_rewards.append(rule_based_reward)

# RL Training Loop
for e in range(episodes):
    state = reset()
    state = np.eye(state_size)[state].reshape(1, -1)
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state = transition(state.argmax(), action)
        reward = rewards[states[state.argmax()]][action]
        done = np.random.rand() < 0.1
        next_state = np.eye(state_size)[next_state].reshape(1, -1)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.replay(batch_size)
    reward_history.append(total_reward)
    cumulative_reward += total_reward
    cumulative_reward_history.append(cumulative_reward)

# Generate and display Q-table
q_table_df = generate_q_table(agent)
print("\nQ-Table:")
print(q_table_df)

# Save Q-table as a CSV file
q_table_df.to_csv("q_table.csv")

# Random Forest Classifier Accuracy
def get_rf_accuracy():
    dataset_path = "heart.csv"
    model = HeartFailurePrediction(dataset_path)
    model.preprocess()
    X_test, y_pred = model.train_model()
    y_test = model.df["target"].iloc[X_test.index]
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

rf_accuracy = get_rf_accuracy()

# Rule-Based Policy Baseline Reward
baseline_reward = evaluate_policy(rule_based_policy, rewards, states, transition, reset, episodes=100)

# Best Recommended Actions for Each State
best_actions = q_table_df.idxmax(axis=1)

# Display Outputs
print(f"\nRandom Forest Classifier Accuracy: {rf_accuracy * 100:.2f}%")
print(f"Rule-Based Policy Baseline Reward: {baseline_reward:.2f}")
print(f"Number of Episodes Run: {episodes}")
print(f"Overall Cumulative Reward: {cumulative_reward:.2f}")
print("\nBest Recommended Actions for Each State:")
for state, action in best_actions.items():
    print(f"  State: {state}, Best Action: {action}")

# Plot Original RL Graph
plt.figure(figsize=(10, 6))
plt.plot(cumulative_reward_history, label="Cumulative Reward", color="orange")
plt.title("Cumulative Reward Over Episodes", fontsize=16)
plt.xlabel("Episodes", fontsize=12)
plt.ylabel("Cumulative Reward", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Prepare Data for Bar Chart Comparison
final_rl_reward = cumulative_reward_history[-1]
final_rule_based_reward = np.sum(rule_based_rewards)

# Plot Bar Chart Comparison
plt.figure(figsize=(8, 6))
bars = plt.bar(["Rule-Based Policy", "RL Policy"], [final_rule_based_reward, final_rl_reward], color=["green", "blue"])
plt.title("Comparison of Final Cumulative Rewards", fontsize=16)
plt.ylabel("Cumulative Reward", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.bar_label(bars, fmt="%.2f", padding=3)
plt.show()
