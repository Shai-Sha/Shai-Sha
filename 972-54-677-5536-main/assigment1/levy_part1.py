#!/usr/bin/env python3

import numpy as np
import sys
from levy_perceptron import Perceptron


def testboolpair(p, I):
    I = np.array(I)
    sys.stdout.write(str(I.astype('bool')) + ' => ')
    print(p.test(I))

def boolpercep(targs, label):

    print(label + ':')

    p = Perceptron(2, 1)

    nptargs = np.zeros((4,1))
    for k in range(4):
        nptargs[k] = targs[k]

    p.train(np.array([[0,0],[0,1],[1,0],[1,1]]), nptargs, niter=10)
    
    testboolpair(p, [0,0])
    testboolpair(p, [0,1])
    testboolpair(p, [1,0])
    testboolpair(p, [1,1])

    print('\n')
 
if __name__ == '__main__':

    boolpercep([0, 1, 1, 1], 'OR')
    boolpercep([0, 0, 0, 1], 'AND')
    boolpercep([1, 1, 1, 0], 'NAND')
    boolpercep([1, 0, 0, 0], 'NOR')
    boolpercep([0, 1, 1, 0], 'XOR')



