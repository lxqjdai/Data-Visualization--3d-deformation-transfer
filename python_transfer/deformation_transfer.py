import v4_normal as v4
import build_elementary_cell as build_e
import numpy as np
import scipy.sparse.linalg
import scipy.sparse
import python2matlab
import scipy.io
def deformation_transfer(vs,fs,vt,ft,vs2,fs2,corres):
    print('Start deformation transfer...')
    lenfs=len(fs)
    lenft=len(ft)
    SD=[]
    ts,ns,vs4,fs4=v4.v4_normal(vs,fs)
    ts2,ns2,vs42,fs42=v4.v4_normal(vs2,fs2)
    tt,nt,vt4,ft4=v4.v4_normal(vt,ft)

    for i in range(lenfs):
        SD.append(np.matmul(ts2[i],np.linalg.inv(ts[i])))#矩阵除法
        
    E=build_e.build_elementary_cell(tt,lenft)

    n_corres = 0
    n_non_corres=0
    #print(corres)
    for i in range(len(corres)):
        if len(corres[i]):
            n_corres += len(corres[i])
        else:
            n_non_corres+=1

    print(n_corres)
    print(n_non_corres)
    I=np.zeros([9*(n_corres+n_non_corres)*4,3])
    C=np.zeros([9*(n_corres+n_non_corres),1])

    print('Transfer deformation')

    offset=0
    offset2=0

    for i in range(lenft):
        if len(corres[i]):
            lencor=len(corres[i])
            cor=corres[i]
        else:
            lencor=0
            cor=corres[i]        
        U=ft4[i]
        if lencor:
            for j in range(lencor):
                for k in range(3):
                    row=np.matlib.repmat(np.array([1,2,3])+offset+j*3*3+k*3,4,1)
                    col1=np.matlib.repmat((U-1)*3+k+1,3,1)
                    col1=np.transpose(col1)
                    val1=np.transpose(E[i])

                    a=np.matlib.reshape(row,12,1)
                    b=np.matlib.reshape(col1,12,1)
                    c=np.matlib.reshape(val1,12,1)
                    I[np.arange(0,12)+offset2+j*3*3*4+k*3*4]=np.transpose(np.vstack((a,b,c)))

                C[np.arange(0,9)+offset+j*9,0]=np.matlib.reshape(np.transpose(SD[int(cor[j-1])-1]),9,1)
            offset=offset+3*3*lencor
            offset2=offset2+3*3*lencor*4
        else:
            for k in range(3):
                row=np.matlib.repmat(np.array([1,2,3])+offset+k*3,4,1)
                col1=np.matlib.repmat((U-1)*3+k+1,3,1)
                col1=np.transpose(col1)
                val1=np.transpose(E[i])
                a=np.matlib.reshape(row,12,1)
                b=np.matlib.reshape(col1,12,1)
                c=np.matlib.reshape(val1,12,1)

                I[np.arange(0,12)+offset2+k*3*4]=np.transpose(np.vstack((a,b,c)))
            C[np.arange(0,9)+offset,0]=np.matlib.reshape(np.eye(3),9,1)
            offset=offset+3*3
            offset2=offset2+3*3*4
    #return I
    scipy.io.savemat("./IandC2",{'I':I,'C':C})
    x=python2matlab.matlabsolver2()
    x=np.reshape(x,[3,int(len(x)/3)],order='F')
    x=np.transpose(x)
    x=x[0:len(vt),:]
    _,nx,_,_=v4.v4_normal(x,ft)

    print('Finish transfer.')
    return x,nx

