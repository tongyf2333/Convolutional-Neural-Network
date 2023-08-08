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
    
def check():
    input=np.zeros((1,1,4,4))
    input[0,0,0,0]=15
    input[0,0,0,1]=3
    input[0,0,0,2]=7
    input[0,0,0,3]=10
    
    input[0,0,1,0]=1
    input[0,0,1,1]=6
    input[0,0,1,2]=9
    input[0,0,1,3]=2
    
    input[0,0,2,0]=0
    input[0,0,2,1]=13
    input[0,0,2,2]=4
    input[0,0,2,3]=14
    
    input[0,0,3,0]=11
    input[0,0,3,1]=5
    input[0,0,3,2]=12
    input[0,0,3,3]=8
    
    a=maxpool(input,2)
    
    b=a.forward(input)
    
    gradient=np.zeros((1,1,2,2))
    
    gradient[0,0,0,0]=1
    gradient[0,0,0,1]=2
    gradient[0,0,1,0]=3
    gradient[0,0,1,1]=4
    
    c=a.backward(gradient)
    
    print(b)
    print(c)
    
#check()