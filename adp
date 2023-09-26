model = AdaBoostClassifier(n_estimators=num_trees,
random_state=seed)
results = model_selection.cross_val_score(model, X, Y)
print(results.mean())
Output:
9. Implement passive reinforcement learning algorithm based on
adaptive dynamic programming (ADP) for the 3 by 4 world
problem
REQUIREMENT: (T.npy) file
Code:
import numpy as np
def return_state_utility(v, T, u, reward, gamma):
action_array = np.zeros(4)
for action in range(0, 4):
action_array[action] = np.sum(np.multiply(u, np.dot(v,
T[:,:,action])))
return reward + gamma * np.max(action_array)
def main():
v = np.array([[0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0,
1.0, 0.0, 0.0, 0.0]])
T = np.load("C:\Ai\T.npy")
u = np.array([[0.812, 0.868, 0.918, 1.0,
0.762, 0.0, 0.660, -1.0,
0.705, 0.655, 0.611, 0.388]])
reward = -0.4
gamma = 1.0
utility_11 = return_state_utility(v, T, u, reward, gamma)
print("Utility of state (1,1): " + str(utility_11))
main()
