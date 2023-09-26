import numpy as np
from itertools import islice
import matplotlib.pyplot as plt

class digit:
    def __init__(self,n,m):
        self.n=n
        self.m=m
        self.w=2*np.random.random((n+1,m))-1
        self.data=[[]]  #16 lines of data that reprisent a digit
        self.I=[]       #one current input array updated after getdata runs
        self.T2=[]      #one current target array updated after getdata runs
        self.batchI=[[]]
        self.batchT=[[]]
        self.filename=""

    def __str__(self):
        return f"A perceptron with {self.n} inputs and {self.m} outputs\n{self.w}"

    def test(self,I):
        #print(f"w={self.w}")
        I=np.append(I,1.)
        #print(f"I={I}")
        IdotW= np.dot(I,self.w)
        #print (f"I*W={IdotW}") 
        return IdotW>np.zeros(len(IdotW))

    def getdata(self,i, filename):
        with open(filename, 'r') as infile:
            self.data = [line for line in infile][i*16:i*16+16]
                
            Arr2D=np.fromstring(self.data[1], dtype=float, sep=' ')
            for line in self.data[2:15]:
                newLine=np.fromstring(line, dtype=float, sep=' ')
                #print(newLine)
                Arr2D=np.vstack((Arr2D,newLine))
            self.I=Arr2D.ravel()
            #get target assuming the array is 0010000000
            two=np.fromstring(self.data[15], dtype=int, sep=' ')[2]
            self.T2=[two, int(not two)]
        infile.close()
        return Arr2D

    def show(self,i):
        Arr2D=self.getdata(i,self.filename)
        plt.imshow(Arr2D, cmap="Greys")
        plt.title(self.data[0])
        plt.show()

    def createbatch(self, filename):
        self.filename=filename
        self.getdata(0, filename)
        self.batchI=self.I
        self.batchT=self.T2
        for i in range(2499):
            self.getdata(i+1,filename)
            self.batchI=np.vstack((self.batchI,self.I))            
            self.batchT=np.vstack((self.batchT,self.T2))
            if i%200==0 and i!=0:
                print(f"{i}/2500 batch creating completed")

    
    def train(self, I, T):

        # Number of patterns p is the number of rows (length) of I
        p = len(I)

        # Augment inputs with a column of 1s for biases
        I = np.hstack(   (I, np.ones((p,1)))  )

        # Train
        for i in range(1000):

            # Set up for batch learning
            deltaw = np.zeros(self.w.shape)
        
            for j in range(p):

                Oj = (np.dot(I[j], self.w)>0)

                Dj = T[j] - Oj

                deltaw += np.outer(I[j], Dj)
                
            # Update weights with batch learning
            self.w += deltaw/p
            if i%100==0 and i!=0:
                print(f"{i}/1000 train iteration completed")

