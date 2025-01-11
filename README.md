Heart Failure Prediction and Drug Recommendation Using Reinforcement Learning
Overview
This project combines machine learning and reinforcement learning to address two key objectives:

Heart Failure Prediction: Using a Random Forest classifier, the project predicts the likelihood of heart failure based on patient data.
Drug Recommendation System: A Deep Q-Learning agent recommends personalized drug treatments or no treatment, based on the state of the patient, optimizing rewards for treatment outcomes.
Features
Reinforcement Learning:
Utilizes a Deep Q-Learning (DQL) agent for drug recommendations.
Includes a rule-based policy for comparison.
Simulates patient state transitions and evaluates policies using realistic reward structures.
Heart Failure Prediction:
Employs Random Forest for accurate heart failure prediction.
Preprocesses patient data, including feature scaling and one-hot encoding for categorical variables.
Q-Table Visualization: Generates and saves a Q-Table summarizing optimal actions for each patient state.
Graphical Comparisons: Visualizes cumulative rewards and policy performance.
Dataset
The project uses a heart failure dataset (heart.csv) with the following features:

age: Age of the patient.
sex: Gender (binary).
cp: Chest pain type (categorical: 0-3).
trestbps: Resting blood pressure.
chol: Serum cholesterol level (mg/dl).
fbs: Fasting blood sugar > 120 mg/dl.
restecg: Resting electrocardiographic results.
thalach: Maximum heart rate achieved.
exang: Exercise-induced angina.
oldpeak: ST depression induced by exercise.
slope: Slope of the peak exercise ST segment.
ca: Number of major vessels colored by fluoroscopy.
thal: Thalassemia status (normal, fixed defect, reversible defect).
target: Heart failure label (binary).
Methods
Reinforcement Learning
Environment: Four patient states (young_low_risk, old_high_risk, mid_age_mid_risk, no_risk) and four actions (Aspirin, Warfarin, Rosuvastatin, no_treatment).
Reward Structure: Custom rewards based on the action's suitability for the patient state.
DQL Agent:
Neural network architecture: Fully connected layers with ReLU activations.
Epsilon-greedy policy for action selection.
Experience replay for learning from past experiences.
Heart Failure Prediction
Preprocessing: Scaling numerical features and one-hot encoding categorical variables.
Random Forest Classifier: Trains a model to predict heart failure with high accuracy.
How to Run
Prerequisites
Python 3.8+
Required libraries: torch, numpy, pandas, scikit-learn, matplotlib.

Steps:
Steps
1.Clone the repository:
bash
Copy code
git clone https://github.com/gokul200902/ML-RL-heart-failure-prediction-and-drug-recommendation.git
cd ML-RL-heart-failure-prediction-and-drug-recommendation
2.Install dependencies:
bash
Copy code
pip install -r requirements.txt
3.Run the main script:
bash
Copy code
python main.py


Results
Random Forest Accuracy: Demonstrates high performance in predicting heart failure.
Policy Performance:
Rule-based policy: Provides a baseline for comparison.
RL policy: Optimized actions to maximize rewards over episodes.
Q-Table: Saved as q_table.csv, summarizing the best actions for each state.
Visualizations
Cumulative rewards graph comparing RL and rule-based policies.
Bar chart comparing final rewards of the two policies.
Future Work
Enhance the RL agent with more advanced algorithms (e.g., Double DQN).
Incorporate larger datasets to improve generalizability.
Deploy the model as a web application for real-world use.
Contribution
Feel free to contribute by creating issues or submitting pull requests.

