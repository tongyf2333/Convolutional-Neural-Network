import numpy as np

class maxpool:
    def __init__(self,input,siz):
        self.input=input
        self.siz=siz
        
    def forward(self,input):
        self.input=input
        batch,channel,H,W=input.shape
        self.x_reshaped=input.reshape(batch,channel,H//self.siz,self.siz,W//self.siz,self.siz)
        out=self.x_reshaped.max(axis=3).max(axis=4)
        self.out=out
        return self.out

    def backward(self,gradient):
        dx_reshaped=np.zeros_like(self.x_reshaped)
        batch,channel,H,W=self.input.shape
        out_newaxis=self.out.reshape(batch,channel,H//self.siz,1,W//self.siz,1)
        #out_newaxis=self.out[:,:,:,np.newaxis,:,np.newaxis]
        mask=(self.x_reshaped==out_newaxis)
        dout_newaxis=gradient.reshape(batch,channel,H//self.siz,1,W//self.siz,1)
        #dout_newaxis=gradient[:,:,:,np.newaxis,:,np.newaxis]
        dout_broadcast,_=np.broadcast_arrays(dout_newaxis,dx_reshaped)
        dx_reshaped[mask]=dout_broadcast[mask]
        dx_reshaped/=np.sum(mask,axis=(3, 5),keepdims=True)
        dx=dx_reshaped.reshape(self.input.shape)
        return dx
    
