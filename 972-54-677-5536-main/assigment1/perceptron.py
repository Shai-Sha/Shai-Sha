import numpy as np
class Perceptron:
        def __init__(self,n,m): #n input, m output
                self.n=n
                self.m=m
                #self.w=0.1*np.random.random((n+1,m))-0.05
                self.w=2*np.random.random((n+1,m))-1
                self.I=[]
        def __str__(self):
                return f"A perceptron with {self.n} inputs and {self.m} outputs\n{self.w}"

        def test(self,I):
                print(f"w={self.w}")
                I=np.append(I,1.)
                print(f"I={I}")
                IdotW= np.dot(I,self.w)
                print (f"I*W={IdotW}") 
                return IdotW>np.zeros(len(IdotW))
        
        def train(self,I,T,w): #,Pattern, Target):
                I=np.array([[0,0],[1,0],[0,1],[1,1]])
                T=np.array([[0,0],[0,1],[0,1],[1,1]])  #[AND,OR]
 
                
                X0 = np.ones(len(I))
                self.I=np.hstack((I,np.reshape(X0,(-1,1))))
                #print (f"Pattern with bias:\n{I}")
                print (f"ini w:\n{self.w}\n")
   
                for _ in range(3):
                        dw=np.zeros(self.w.shape)
                        for i in range(len(self.I)):
                                Oi=np.dot(self.I[i],self.w)
                                print (f"OUT[{i}]:{Oi}\n")
                                Di=T[i]-Oi
                                print (f"Di[{i}]:{Di}\n")
                                #print (f"Oi[{i}]:{Oi}\n")
                                Di=np.outer(Di,np.ones(len(self.I[i]))).T
                                dw+=np.dot(self.I[i],Di)*0.01
                        self.w+=dw/len(self.I)
                print (f"w:\n{self.w}\n")
                return self.w
                """O=np.dot(P,self.w)>0
                        #D=np.outer(np.ones(self.m),Target)-O
                        D=Target-O
                        dw=np.dot(P.T,D)
                        print (f"P.T={P.T}")
                        print (f"D={D}")
                        print (f"dw={dw}")
                        self.w+=dw/n
                print (f"O={O}")
                print (f"D={D}")
                print (f"Testing AND on [0,0]{test([0,0])}")"""
                
