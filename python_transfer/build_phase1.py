from scipy.sparse import csr_matrix
import scipy.sparse
import numpy as np
import scipy.io
def build_phase1(Adj_idx,E,fs4,vt4,ws,wi,marker):
    n_adj=Adj_idx.size
    len_col=np.max(fs4)
    I1=np.zeros([9*n_adj*4,3])
    I2=np.zeros([9*n_adj*4,3])
    I3=np.zeros([9*len(fs4)*4,3])
    C1=np.zeros([9*n_adj,1])
    C2=wi*np.matlib.reshape(np.transpose(np.matlib.repmat(np.matlib.reshape(np.eye(3),9,1),len(fs4),1)),9*len(fs4),1)
    print('buiilding phase 1')
    for i in range(len(fs4)):
        for j in range(3):
            if Adj_idx[i,j]:
                constid=np.zeros([2,4])
                for k in range(3):
                    if np.sum(marker[:,0:1]==fs4[i,k]):
                        constid[0,k]=(k+1)*np.sum(marker[:,0:1]==fs4[i,k])
                    if np.sum(marker[:,0:1]==fs4[Adj_idx[i,j]-1,k]):
                        constid[1,k]=(k+1)*np.sum(marker[:,0:1]==fs4[Adj_idx[i,j]-1,k])
                U1=fs4[i]
                U2=fs4[int(Adj_idx[i,j])-1]
                for k in range(3):
                    row=np.matlib.repmat(np.array([1,2,3])+i*3*3*3+j*3*3+k*3,4,1)
                    col1=np.matlib.repmat((U1-1)*3+k+1,3,1)
                    col1=np.transpose(col1)
                    val1=ws*np.transpose(E[i])
                    if np.sum(constid[0]):
                        C1[np.array([1,2,3])+i*3*3*3+j*3*3+k*3-1]=C1[np.array([1,2,3])+i*3*3*3+j*3*3+k*3-1]-np.transpose(val1[np.nonzero(constid[0]>0)])*np.transpose(vt4[marker[np.nonzero((marker==U1[constid[0]>0])+0),1][0]-1,k])
                    val1[constid[0]>0]=0
                    col2=np.transpose(np.matlib.repmat((U2-1)*3+k+1,3,1))
                    val2=np.transpose(-ws*E[Adj_idx[i,j]-1])
                    if np.sum(constid[1]):
                        C1[np.array([1,2,3])+i*3*3*3+j*3*3+k*3-1]=C1[np.array([1,2,3])+i*3*3*3+j*3*3+k*3-1]-np.transpose(val2[np.nonzero(constid[1]>0)])*np.transpose(vt4[marker[np.nonzero((marker==U2[constid[1]>0])+0),1][0]-1,k])
                    a=np.matlib.reshape(row,12,1)
                    b=np.matlib.reshape(col1,12,1)
                    c=np.matlib.reshape(val1,12,1)
                    I1[np.arange(0,12)+i*3*3*3*4+j*3*3*4+k*3*4]=np.transpose(np.vstack((a,b,c)))
                    b=np.matlib.reshape(col2,12,1)
                    c=np.matlib.reshape(val2,12,1)
                    I2[np.arange(0,12)+i*3*3*3*4+j*3*3*4+k*3*4]=np.transpose(np.vstack((a,b,c)))   
    I1=I1[I1[:,0]>0]
    I2=I2[I2[:,0]>0]
    #M1=csr_matrix((I1[:,2],(I1[:,0]-1,I1[:,1]-1)),shape=(9*n_adj,3*len_col))
    #M2=csr_matrix((I2[:,2],(I2[:,0]-1,I2[:,1]-1)),shape=(9*n_adj,3*len_col))
    #M3=M1+M2

    for i in range(len(fs4)):
        U1=fs4[i]
        for k in range(3):
            row=np.matlib.repmat(np.array([1,2,3])+i*3*3+k*3,4,1)
            col1=np.matlib.repmat((U1-1)*3+k+1,3,1)
            col1=np.transpose(col1)
            val1=wi*np.transpose(E[i])
            a=np.matlib.reshape(row,12,1)
            b=np.matlib.reshape(col1,12,1)
            c=np.matlib.reshape(val1,12,1)
            I3[np.arange(0,12)+i*3*3*4+k*3*4]=np.transpose(np.vstack((a,b,c)))
    #M4=csr_matrix((I3[:,2],(I3[:,0]-1,I3[:,1]-1)),(int(np.max(I3[:,0])),int(np.max(I3[:,1]))))
    C=np.hstack((np.squeeze(C1),C2))
    #M=scipy.sparse.vstack((M3,M4))
    #print(M)
    scipy.io.savemat('./IandC',{'I1':I1,'I2':I2,'I3':I3,'C':C})
    return I1,I2,I3,C