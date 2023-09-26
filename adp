import numpy as np
def policy(v, t, u, r, g):
    action = np.zeros(4)
    for a in range(0, 4):
        action[a] = np.sum(np.multiply(u, np.dot(v, t[:,:,a])))
    return r + g * np.max(action)

v = np.array([[0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 
               1.0, 0.0, 0.0, 0.0]])
t = np.load("t.npy") 
u = np.array([[0.812, 0.868, 0.918,   1.0,
               0.762,   0.0, 0.660,  -1.0,
               0.705, 0.655, 0.611, 0.388]])
r = -0.4
g = 1.0
utility = policy(v, t, u, r, g)
print("Utility of state (1,1): " + str(utility))
