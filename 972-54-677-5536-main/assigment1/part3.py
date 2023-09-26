import matplotlib.pyplot as plt
import numpy as np
# Harrison's one-line solution!
x1, x2, y, z = np.loadtxt('assign1_data.txt', skiprows = 2, unpack = True) #a very nice one liner
A = np.vstack([x1, x2, np.ones(len(x1))]).T

w1, w2, b = np.linalg.lstsq(A, y, rcond=None)[0]
success=0
for i in range(len(x1)): 
    if(((x1[i]*w1+x2[i]*w2+b)>0)==z[i]):
        success+=1

print((int)(success/len(x1)*100))
