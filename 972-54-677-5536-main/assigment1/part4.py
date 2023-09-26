import matplotlib.pyplot as plt
import numpy as np
# Harrison's one-line solution!
x1, x2, y, z = np.loadtxt('assign1_data.txt', skiprows = 2, unpack = True) #a very nice one liner
A = np.vstack([x1, x2, np.ones(len(x1))]).T

w1, w2, b = np.linalg.lstsq(A, y, rcond=None)[0]
success=0

w=np.array([w1, w2, b])

result=np.dot(A,w)>0
for i in range(len(result)):
        if(result[i]==z[i]):
                success+=1
        
print (f"Success with  lstsq is: {success}")
print (f"w={w}")

#w=np.random.randn(2+1) # 1 for bias
#instead of using lstsq I used learning
def learn(x1,x2,y,w):   
        for _ in range(1000):
                dw=np.zeros(w.shape)
                for i in range(len(x1)):
                        I=[x1[i],x2[i],1]
                        Oi=np.dot(I,w)
                        Di=y[i]-Oi
                        dw+=np.dot(I,Di)
                w+=dw/len(x1)
        return w
for batch in [25,50,75]: #Simon loop is more elegant
        w=learn(x1[:batch],x2[:batch],y[:batch],np.array([0.,0.,0.]))
        A = np.vstack([x1[batch:], x2[batch:], np.ones(100-batch)]).T
        result=(np.dot(A,w)>0)
        success=sum(result==z[batch:])#check learning against the rest data
        print (f"Success (Perceptron) with batch of {batch}  and {iter} iteration is: {success/(100-batch)*100:.2f}%")
        print (f"w={w}")
