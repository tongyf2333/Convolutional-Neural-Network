import numpy as np
from conv_cython import dim

class conv:
    def __init__(self,input,w,b,stride=1):
        self.input=input
        self.w=w
        self.b=b
        self.stride=stride
        self.dw=np.zeros_like(w)
        self.db=np.zeros_like(b)
        
    def forward(self,input):
        self.input=input
        batch,channel,H,W=input.shape
        core,_,h,w=self.w.shape
        stride=self.stride
        dimx=(H-h)//stride+1
        dimy=(W-w)//stride+1
        
        shape=(channel,h,w,batch,dimx,dimy)
        strides=(H*W,W,1,channel*H*W,stride*W,stride)
        strides=input.itemsize*np.array(strides)
        x_stride=np.lib.stride_tricks.as_strided(input,shape=shape,strides=strides)
        x_cols=np.ascontiguousarray(x_stride)
        x_cols.shape=(channel*h*w,batch*dimx*dimy)
        
        res=self.w.reshape(core,-1).dot(x_cols)+self.b.reshape(-1,1)
        res.shape=(core,batch,dimx,dimy)
        out=res.transpose(1,0,2,3)
        out=np.ascontiguousarray(out)
        self.out=out
        self.x_cols=x_cols
        return out
    
    def backward(self,gradient):
        batch,channel,H,W=self.input.shape
        core,_,h,w=self.w.shape
        _,_,dimx,dimy=self.out.shape
        
        self.db=np.sum(gradient,axis=(0,2,3))
        self.db=self.db.reshape(-1,1)
        gradient_reshaped=gradient.transpose(1,0,2,3).reshape(core,-1)
        self.dw=gradient_reshaped.dot(self.x_cols.T).reshape(self.w.shape)
        
        dx_cols=self.w.reshape(core,-1).T.dot(gradient_reshaped)
        dx_cols.shape=(channel,h,w,batch,dimx,dimy)
        tmp=np.zeros((batch,channel,H,W))
        dx=dim(dx_cols,tmp,batch,channel,H,W,h,w,self.stride)
        
        return dx
    
def check():
    input=np.zeros((1,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                input[0,i,j,k]=i*9+j*3+k
                
    kernel=np.zeros((1,3,2,2))
    for i in range(3):
        for j in range(2):
            for k in range(2):
                kernel[0,i,j,k]=i*4+j*2+k
                
    a=conv(input,kernel,np.zeros((1,1)))
    
    b=a.forward(input)
    
    gradient=np.zeros((1,1,2,2))
    for i in range(2):
        for j in range(2):
            gradient[0,0,i,j]=i*2+j+1
            
    c=a.backward(gradient)
    
    print(gradient)
    print(c)
    
#check()