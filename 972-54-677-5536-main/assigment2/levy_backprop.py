#!/usr/bin/env python3

import numpy as np
import sys
import pickle

class Backprop(object):

    def __init__(self, n, h, m, weightscale=1): 

        self.wih = self._makeweights(n, h, weightscale)
        self.who = self._makeweights(h, m, weightscale)

    def _makeweights(self, p, q, weightscale):

        return np.random.randn(p+1,q) * weightscale

    def __str__(self):

        return '%d X %d X %d' % (self.wih.shape[0], self.wih.shape[1]-1, self.who.shape[1]-1)

    def _f(self, x):
        return 1 / (1 + np.exp(-x))


    def _df(self, x):
        y = self._f(x)
        return y * (1-y)

    def setWeights(self, wih, who):

        self.wih = wih
        self.who = who

    def getWeights(self):

        return self.wih, self.who

    def test(self, Ij):

        hnet = np.dot(np.append(Ij,1),self.wih)
        h = self._f(hnet)

        onet = np.dot(np.append(h,1),self.who)
        o = self._f(onet)

        return o

    def getHidden(self, Ij):

        hnet = np.dot(np.append(Ij,1),self.wih)
        h = self._f(hnet)

        return h

    def rmserr(self, I, T):

        sumerr = 0

        for Ij,Tj in zip(I,T):

            hnet = np.dot(np.append(Ij,1),self.wih)
            h = self._f(hnet)

            onet = np.dot(np.append(h,1),self.who)
            o = self._f(onet)

            eo = Tj - o
            do = eo * self._df(onet)

            eh = np.dot(do, self.who.T)[:-1]
            dh = eh * self._df(hnet)

            sumerr += sum(eo**2)
        
        return np.sqrt(sumerr / (len(I)*self.who.shape[1]))

    def train(self, I, T, eta=.05, mu=0.0, niter=1000, report=1):

        # Create empty RMS error array
        rmserr = np.zeros(niter)

        # Create previous-weight-change matrices for momentum
        dwih_prev = np.zeros(self.wih.shape)
        dwho_prev = np.zeros(self.who.shape)

        for i in range(niter):

            # Set up for batch learning
            dwih = np.zeros(self.wih.shape)
            dwho = np.zeros(self.who.shape)

            # sum-squared error
            sumerr = 0

            for Ij,Tj in zip(I,T):

                hnet = np.dot(np.append(Ij,1),self.wih)
                h = self._f(hnet)

                onet = np.dot(np.append(h,1),self.who)
                o = self._f(onet)

                eo = Tj - o
                do = eo * self._df(onet)

                # This is back-prop!  Note that we drop the final, unusable
                # error value associated with the bias unit
                eh = np.dot(do, self.who.T)[:-1]

                dh = eh * self._df(hnet)

                dwih += np.outer(np.append(Ij,1), dh)
                dwho += np.outer(np.append(h,1),  do)

                sumerr += sum(eo**2)

            # Average weight changes over entire batch
            dwih /= len(I)
            dwho /= len(I)

            # Update weights with batch learning and momentum
            self.wih += eta * dwih + mu * dwih_prev
            self.who += eta * dwho + mu * dwho_prev

            # Track previous weight changes for momentum
            dwih_prev = dwih
            dwho_prev = dwho
            
            # Report RMS error periodically
            rmserr[i] = np.sqrt(sumerr / (len(I)*self.who.shape[1]))
            if i%report == 0:
                print('%d/%d: %f' % (i, niter, rmserr[i]))

        return rmserr

    def save(self, filename):

        pickle.dump([self.wih,self.who], open(filename, 'wb'))

    def load(self, filename):

        self.wih, self.who = pickle.load(open(filename, 'rb'))

