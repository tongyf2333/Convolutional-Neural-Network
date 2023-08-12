from conv import conv
from maxpool import maxpool
from flatten import flatten
from fc import fc
from relu import relu
from bn import bn
from bn_linear import bn_linear
from softmax import softmax
import numpy as np

def loss(x,y):
    return -1*np.sum(np.multiply(y,np.log(x)),axis=(0,1))

class CNN:
    def __init__(self,rate):
        self.layers=[]
        self.rate=rate
    
    def build(self,batch):
        layer1=np.zeros((batch,1,28,28))
        kernel1=np.random.randn(16,1,3,3)*np.sqrt(2.0/(1*28*28+16*26*26))
        b1=np.zeros((16,1))
        self.layers.append(conv(layer1,kernel1,b1))
        
        layer2=np.zeros((batch,16,26,26))
        self.layers.append(bn(layer2))
        self.layers.append(relu(layer2))
        self.layers.append(maxpool(layer2,2))
        
        layer3=np.zeros((batch,16,13,13))
        kernel2=np.random.randn(32,16,4,4)*np.sqrt(2.0/(16*13*13+32*10*10))
        b2=np.zeros((32,1))
        self.layers.append(conv(layer3,kernel2,b2))
        
        layer4=np.zeros((batch,32,10,10))
        self.layers.append(bn(layer4))
        self.layers.append(relu(layer4))
        self.layers.append(maxpool(layer4,2))
        
        layer5=np.zeros((batch,32,5,5))
        self.layers.append(flatten(layer5))
        
        layer6=np.zeros((800,batch))
        kernel3=np.random.randn(10,800)*np.sqrt(2.0/810)
        b3=np.zeros((10,1))
        self.layers.append(fc(layer6,kernel3,b3))
        
        layer7=np.zeros((10,batch))
        self.layers.append(bn_linear(layer7))
        self.layers.append(softmax(layer7))
        
    def train_batch(self,x,y):
        num=len(self.layers)
        input=x
        for i in range(num):
            output=self.layers[i].forward(input)
            input=output
        
        i=num-1
        gradient=y
        while i>=0 :
            gradient=self.layers[i].backward(gradient)
            i-=1
            
    def test(self,X,Y):
        X=np.arctan(X)*(2/np.pi)
        batch=X.shape[0]
        res=0
        num=len(self.layers)
        input=X
        for j in range(num):
            output=self.layers[j].forward(input)
            input=output
        for i in range(batch):
            if input[:,i].argmax(0)==Y[:,i].argmax(0) :
                res+=1
        L=loss(input,Y)
        print(L/batch)
        print("accuracy:{}".format(res/batch))
        
        
    def train(self,X,Y):
        X=np.arctan(X)*(2/np.pi)
        batch=X.shape[0]
        self.train_batch(X,Y)
        
        num=len(self.layers)
        for i in range(num):
            if type(self.layers[i])==conv :
                self.layers[i].w-=self.layers[i].dw*self.rate/batch
                self.layers[i].b-=self.layers[i].db*self.rate/batch
                self.layers[i].dw=np.zeros_like(self.layers[i].w)
                self.layers[i].db=np.zeros_like(self.layers[i].b)
            if type(self.layers[i])==fc :
                self.layers[i].w-=self.layers[i].dw*self.rate/batch
                self.layers[i].b-=self.layers[i].db*self.rate/batch
                self.layers[i].dw=np.zeros_like(self.layers[i].w)
                self.layers[i].db=np.zeros_like(self.layers[i].b)
            if type(self.layers[i])==bn :
                self.layers[i].gamma-=self.layers[i].dgamma*self.rate/batch
                self.layers[i].beta-=self.layers[i].dbeta*self.rate/batch
                self.layers[i].dgamma=np.zeros_like(self.layers[i].gamma)
                self.layers[i].dbeta=0
            if type(self.layers[i])==bn_linear :
                self.layers[i].gamma-=self.layers[i].dgamma*self.rate/batch
                self.layers[i].beta-=self.layers[i].dbeta*self.rate/batch
                self.layers[i].dgamma=np.zeros_like(self.layers[i].gamma)
                self.layers[i].dbeta=0