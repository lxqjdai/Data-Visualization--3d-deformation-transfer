import numpy as np
from joblib import Parallel,delayed
def build_adjacency(fs):  #需要确认
    Adj_idx=np.zeros((len(fs),3),dtype=u'int64')
    def Adj(i):
        for j in range(3):
            temp1=np.sum(fs==fs[i,j],1) 
            temp2=np.sum(fs==fs[i,(j+1)%3],1)
            idx=np.nonzero(temp1 & temp2)
            temp=np.array([i]*len(idx[0]))
            if np.sum(idx!=temp):
                Adj_idx[i,j]=idx[0][np.nonzero(((idx!=temp)+0)[0])]+1
    Parallel(n_jobs=6,require='sharedmem')(delayed(Adj)(i) for i in range(len(fs)))
    return Adj_idx