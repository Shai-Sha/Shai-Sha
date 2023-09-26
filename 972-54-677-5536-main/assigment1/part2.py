import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([0, 1, 2, 3])
x2 = np.array([0, 1, 2, 3])*2
y = np.array([-1, 0.2, 0.9, 2.1])
A = np.vstack([x1, x2, np.ones(len(x1))]).T

w1, w2, b = np.linalg.lstsq(A, y, rcond=None)[0]

print(w1,w2,b)
