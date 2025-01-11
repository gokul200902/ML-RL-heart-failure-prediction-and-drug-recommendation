import numpy as np

# Rule-Based Policy
def rule_based_policy(state):
    if state == 0:  # young_low_risk
        return 0  # Drug_1
    elif state == 1:  # old_high_risk
        return 1  # Drug_2
    elif state == 2:  # mid_age_mid_risk
        return 2  # Drug_3
    elif state == 3:  # no_risk
        return 3  # no_treatment

# Evaluate Policies
def evaluate_policy(policy, rewards, states, transition_fn, reset_fn, episodes=100):
    total_rewards = []
    for _ in range(episodes):
        state = reset_fn()
        total_reward = 0
        done = False
        while not done:
            action = policy(state)
            reward = rewards[states[state]][action]
            total_reward += reward
            state = transition_fn(state, action)
            if np.random.rand() < 0.1:  # Termination condition
                done = True
        total_rewards.append(total_reward)
    return np.mean(total_rewards)
