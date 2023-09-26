import numpy as np
import matplotlib.pyplot as plt

from itertools import islice
with open('digits_train.txt', 'r') as infile:
    lines = [line for line in infile][:16]
    Arr2D=newLine=np.fromstring(lines[1], dtype=float, sep=' ')
    for line in lines[2:15]:
        newLine=np.fromstring(line, dtype=float, sep=' ')
        print(newLine)
        
        Arr2D=np.vstack((Arr2D,newLine))
im = plt.imshow(Arr2D, cmap="copper_r")
plt.colorbar(im)
plt.show()
