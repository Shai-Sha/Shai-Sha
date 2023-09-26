#!/usr/bin/env python3

import numpy as np

class Perceptron(object):

    def __init__(self, n, m):

        self.w = 0.001 * np.random.randn(n+1,m)

    def __str__(self):

        return str(self.w)

    def test(self, I):

        return np.dot(np.append(I, 1.0), self.w) > 0

    def train(self, I, T, niter=1000, report=10):

        # Number of patterns p is the number of rows (length) of I
        p = len(I)

        # Augment inputs with a column of 1s for biases
        I = np.hstack(   (I, np.ones((p,1)))  )

        # Train
        for i in range(niter):

            # Set up for batch learning
            deltaw = np.zeros(self.w.shape)
        
            for j in range(p):

                Oj = (np.dot(I[j], self.w)>0)

                Dj = T[j] - Oj

                deltaw += np.outer(I[j], Dj)
                
            # Update weights with batch learning
            self.w += deltaw/p

            if (i+1)%report == 0:
                print('Completed ' + str(i+1) + ' / ' + str(niter) + ' iterations')

