#!/usr/bin/env python3

import numpy as np
import sys
from levy_backprop import Backprop

def testboolpair(p, I):
    I = np.array(I)
    sys.stdout.write(str(I.astype('bool')) + ' => ')
    print(p.test(I))

if __name__ == '__main__':

    p = Backprop(2, 4, 1)

    rmserr = p.train(np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([[0],[1],[1],[0]]), eta=1.0, niter=10000, report=1000)
    
    testboolpair(p, [0,0])
    testboolpair(p, [0,1])
    testboolpair(p, [1,0])
    testboolpair(p, [1,1])
