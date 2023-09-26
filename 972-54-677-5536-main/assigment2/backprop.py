import numpy as np
import matplotlib.pyplot as plt
import pickle

class Backprop:
    def __init__(self,n,h,m):
        self.n=n
        self.m=m
        self.h=h        #the number of hidden units
        self.w=[0,0]
        self.w[0]=np.random.randn(n+1,h)
        self.w[1]=np.random.randn(h+1,m)
        self.data=[[]]  #16 lines of data that reprisent a digit
        self.I=[]       #one current input array updated after getdata runs
        self.T2=[]      #one current target array updated after getdata runs
        self.batchI=[[]]
        self.batchT=[[]]
        self.filename="digits_train.txt"
        #self.confusionMatrix = np.zeros((m, m))

    def __str__(self):
        return f"A backprop with {self.n} inputs, {self.h} hiddens and {self.m} outputs\n{self.w}"

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

##    def updateConfusionMatrix(self,t,o):
##        self.confusionMatrix[t][o]+= 1

    def showConfusion(self):
        confMatrix = np.zeros((self.m, self.m))
        for i in range(0,2500):
            O=self.test(self.batchI[i])
            T=self.batchT[i]
            confMatrix[np.argmax(T)][np.argmax(O)]+=1
        print(confMatrix)


    #testing net with hidden layer
    def test(self,I):
        hnet= np.dot(np.append(I,1.),self.w[0])
        h=self.sigmoid(hnet)    #squash the result of hidden net output
        onet= np.dot(np.append(h,1),self.w[1])
        onet=self.sigmoid(onet)
        return onet

    #read the data of a single digit (1 of 2500) from file
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
        return self.data

    def show(self,i):
        data=self.getdata(i,self.filename)
        Arr2D=np.fromstring(data[1], dtype=float, sep=' ')
        plt.imshow(Arr2D, cmap="Greys")
        plt.title(data[0])
        plt.show()

    def createbatch(self, filename, selectT=2):
        self.filename=filename
        data=self.getdata(0, filename)
        self.batchI=self.I
        if selectT==2:
            two=np.fromstring(self.data[15], dtype=int, sep=' ')[2]
            self.batchT=[two, int(not two)]
        else:
            self.batchT=np.fromstring(self.data[15], dtype=int, sep=' ')
        for i in range(2499):
            data=self.getdata(i+1,filename)
            self.batchI=np.vstack((self.batchI,self.I))
            if selectT==2:
                two=np.fromstring(self.data[15], dtype=int, sep=' ')[2]
                newT=[two, int(not two)]
            else:
                newT=np.fromstring(self.data[15], dtype=int, sep=' ')
            self.batchT=np.vstack((self.batchT,newT))
            if i%250==0:
                print(f"{i}/2500 batch creating completed")

    #eta is the learining rate, I inputs, T targets
    def train(self, I, T, eta=0.05, mu=0.0, niter=1000, report=100):

        # Create empty RMS error array
        rmserr = np.zeros(niter)

        # Number of patterns p is the number of rows (length) of I
        p = len(I)

        # Augment inputs with a column of 1s for biases
        I = np.hstack(   (I, np.ones((p,1)))  )

        # Train
        for i in range(niter):

            # sum-squared error
            sumerr = 0

            # zeros init error
            dwih = np.zeros(self.w[0].shape) # for input hidden weights
            dwho = np.zeros(self.w[1].shape) # for hidden out weights
        

            for j in range(p):
                Hnet = np.dot(I[j], self.w[0])  #I already include bias
                H = self.sigmoid(Hnet)               #squashing function (values from 0 to 1)
                Onet = np.dot(np.append(H,1), self.w[1]) #adding bias to hidden layer
                O = self.sigmoid(Onet)               #squashing again (values from 0 to 1)
                
                eo=T[j]-O                       #The error
                difsig_O = O*(1-O)              #differential of the squation sigmoid function
                d_o = eo*difsig_O               #the error on the output of the hidden layer
                sumerr += sum(eo**2)            #suming error for tracing progress

                #back prop
                difsig_H = H*(1-H)              #differential of the squation sigmoid function
                d_h = np.dot(d_o,self.w[1].T)   #going back calc the error delta (that's why the transpose)
                d_h = d_h[:-1]*difsig_H         #droping the unnescery hiden bias error for backprop error calc

                #summing correction to weights over selected patterns
                dwih += np.outer(I[j], d_h)     
                dwho += np.outer(np.append(H,1), d_o)
                
            # Update weights with batch learning
            self.w[0] += eta*dwih/p
            self.w[1] += eta*dwho/p

            # Report RMS error periodically
            rmserr[i] = np.sqrt(sumerr / (p*self.w[1].shape[1]))
            if i%report == 0:
                print('%d/%d: %f' % (i, niter, rmserr[i]))
                self.showConfusion()
                
        return rmserr


    def save(self, filename):

        pickle.dump([self.w[0],self.w[1]], open(filename, 'wb'))

    def load(self, filename):

        self.w[0], self.w[1] = pickle.load(open(filename, 'rb'))

