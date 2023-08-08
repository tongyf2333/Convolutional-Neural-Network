import numpy as np

class flatten:
    def __init__(self,input):
        self.input=input
        
    def forward(self,input):
        self.input=input
        batch,channel,H,W=input.shape
        self.out=input.reshape(batch,channel*H*W)
        self.out=self.out.T
        return self.out
    
    def backward(self,gradient):
        batch,channel,H,W=self.input.shape
        ans=gradient.reshape(channel*H*W,batch)
        ans=ans.T
        ans=ans.reshape(batch,channel,H,W)
        return ans