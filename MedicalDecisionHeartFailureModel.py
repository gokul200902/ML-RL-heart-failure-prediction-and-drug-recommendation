import numpy as np

# RL Environment
states = ["young_low_risk", "old_high_risk", "mid_age_mid_risk", "no_risk"]
actions = ["Aspirin", "Warfarin", "Rosuvastatin", "no_treatment"]

# Simulated reward structure based on realistic scenarios
rewards = {
    "young_low_risk": [1.0, 0.2, -0.1, -1.0],
    "old_high_risk": [-0.2, 1.0, 0.5, -1.0],
    "mid_age_mid_risk": [0.4, 0.8, 0.2, -1.0],
    "no_risk": [-0.5, -0.5, -0.5, 1.0],
}

# Reset function
def reset():
    return np.random.choice(len(states))

# Transition function
def transition(state, action):
    probabilities = {
        0: [0.6, 0.3, 0.1, 0.0],  # young_low_risk
        1: [0.2, 0.6, 0.2, 0.0],  # old_high_risk
        2: [0.3, 0.4, 0.3, 0.0],  # mid_age_mid_risk
        3: [0.0, 0.0, 0.0, 1.0],  # no_risk (remains in no_risk)
    }
    return np.random.choice(len(states), p=probabilities[state])
