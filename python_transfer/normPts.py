import numpy as np
import numpy.matlib
def normPts(x,m,s):
    n=x.shape[0]
    k=x.shape[1]
    T=np.eye(k+1)
    
    mu=np.mean(x,0)
    T[0:k,k]=0-mu
    temp=np.power(x-np.matlib.repmat(mu,n,1),2)
    temp=np.power(temp,0.5)
    mean_distance=np.mean(np.sum(temp,1))
    
    scale=s/mean_distance
    T=scale*T
    T[k,k]=1

    temp=np.ones(len(x))
    temp = temp[:,np.newaxis]
    
    x=np.hstack((x,temp))
    x=np.transpose(np.matmul(T,np.transpose(x)))
    x=x[:,0:k]
    return x