import numpy as np

class relu:
    def __init__(self,input):
        self.input=input
        
    def forward(self,input):
        self.input=input
        self.out=np.maximum(0,input)
        return self.out
    
    def backward(self,gradient):
        dx,x=None,self.input
        dx=gradient
        dx[x<0]=0
        return dx