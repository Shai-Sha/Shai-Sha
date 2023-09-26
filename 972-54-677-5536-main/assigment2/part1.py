#!/usr/bin/env python3

import numpy as np
import sys
from backprop import Backprop
import datetime

def testboolpair(p, I):
    I = np.array(I)
    sys.stdout.write(str(I.astype('bool')) + ' => ')
    print(p.test(I))

def testXorLearning():
    print("Xor training")
    p = Backprop(2, 3, 1)

    p.train(np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([[0],[1],[1],[0]]), eta=1.0, niter=10000, report=1000)
    
    testboolpair(p, [0,0])
    testboolpair(p, [0,1])
    testboolpair(p, [1,0])
    testboolpair(p, [1,1])

def testDigit2Learning(loadNet):
    print("Digit 2 testing")
    d = Backprop(196, 5, 2)
    if loadNet!=1:
        print("Digit 2 training")
        #read data from train file
        d.createbatch('digits_train.txt') 
        print("End reading data and Start training")

        print("Current Time:",datetime.datetime.now().time())
        #train NN param:input, target, eta, mu, niter, report
        d.train(d.batchI,d.batchT,0.75,0,20000,100) 
        print("End training and start reading testing data")
        d.save("testNNdigit2")
    else:
        d.load("testNNdigit2")
    #read data from test file
    d.createbatch('digits_test.txt')
    print("End reading testing data and start testing")

    #start testing NN
    success=0
    start=500
    end=750
    for i in range(start,end):
        test=d.test(d.batchI[i])
        if test[0] > test[1]:
            success=success+1
        if i%25==0:
            print("i=%d T=(%d,%d) O=(%.2f,%.2f) "%(i,d.batchT[i][0],d.batchT[i][1],test[0],test[1]))
    total=end-start
    #print(f"number of success recognizing digit '2' out of 250 paterns is: {count} - {count/250*100}%")
    print("%d misses / %d = %.2f%%"%(total-success,total,(total-success)/total*100))

    countNo2=0
    for i in range(0,500):
        test=d.test(d.batchI[i])
        if test[0] < test[1]:
            countNo2=countNo2+1
        if i%100==0:
            print("i=%d T=(%d,%d) O=(%.2f,%.2f) "%(i,d.batchT[i][0],d.batchT[i][1],test[0],test[1]))

    for i in range(750,2500):
        test=d.test(d.batchI[i])
        if test[0] < test[1]:
            countNo2=countNo2+1
        if i%100==0:
            print("i=%d T=(%d,%d) O=(%.2f,%.2f) "%(i,d.batchT[i][0],d.batchT[i][1],test[0],test[1]))

    print("%d false positive / 2250 = %.2f%%"%(2250-countNo2,(2250-countNo2)/2250*100))
    return d

def fullMonty10digit(loadNet):
    print("Full Monty 10 Digit testing")
    d = Backprop(196, 10, 2)
    if loadNet!=1:
        print("training...")
        #read data from train file
        d.createbatch('digits_train.txt') 
        print("End reading data and Start training")

        #train NN param:input, target, eta, mu, niter, report
        d.train(d.batchI,d.batchT,0.1,0,10000,100) 
        print("End training and start reading testing data")
        d.save("testNNdigit2")
    else:
        d.load("testNNdigit2")
    #read data from test file
    d.createbatch('digits_test.txt')
    print("End reading testing data and start testing")

    
if __name__ == '__main__':
    testXorLearning()
    #d=testDigit2Learning(0) #1 means load existing NNet
    print("Current Time:",datetime.datetime.now().time())
