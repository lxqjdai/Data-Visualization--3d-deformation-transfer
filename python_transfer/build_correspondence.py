from scipy.spatial import KDTree
import numpy as np
def build_correspondence(VS, FS, VT, FT, maxind, thres):
    import numpy.matlib
    def v4_normal(Vert,Face):
        f1, f2, f3 = Face[:,0],Face[:,1],Face[:,2]
        e1 = Vert[f2-1,:]-Vert[f1-1,:] # index务必减一！
        e2 = Vert[f3-1,:]-Vert[f1-1,:] 

        c=np.matlib.cross(e1,e2,1)
        c_norm=(np.power(c[:,0:1],2)+np.power(c[:,1:2],2)+np.power(c[:,2:3],2))**0.5
        c_norm[c_norm==0]=1
        N=c/np.matlib.repmat(c_norm,1,3)
        v4=Vert[list(f1-1),:]+N
        V=np.vstack((Vert,v4))
        F=Face
        temp=len(Vert)+numpy.arange(1,len(F)+1)
        temp=temp[:,np.newaxis]
        F=np.hstack((F,temp))
        T=[]
        for i in range(0,len(F)):
            x=V[F[i,1]-1,]-V[F[i,0]-1,]
            y=V[F[i,2]-1,]-V[F[i,0]-1,]
            z=V[F[i,3]-1,]-V[F[i,0]-1,]
            t=np.array([x,y,z])
            T.append(np.transpose(t))
        return T,N,V,F
    # conpute v4 and normalize other varibale
    TS,NS,VS4,FS4= v4_normal(VS, FS) 
    TT,NT,VT4,FT4 = v4_normal(VT, FT)
    VS_C = np.zeros([FS.shape[0],3])
    VT_C = np.zeros([FT.shape[0],3])
    # take the mean vertices of every triangles in target face
    for i in range(0,FT.shape[0]):
        VT_C[i,:] = np.mean(VT[FT[i,:]-1,:],0)
    # take the mean vertices of every triangles in source face
    for i in range(0,FS.shape[0]):
        VS_C[i,:] = np.mean(VS[FS[i,:]-1,:],0)
    # build kd trees
    S_tree = KDTree(VS_C)
    T_tree = KDTree(VT_C)
    corres1 = np.zeros([len(FT),maxind]) # set corresponding matrix
    corres2 = np.zeros([len(FT),maxind]) # set corresponding matrix
    templength = 0 # temp len of the num of  correponds
    len_ = 0 # initial value of corresponds
    rowlen = -1
    for i in range(FS.shape[0]): # find corresponds from source
        corresind = T_tree.query(VS_C[i,:],k=maxind,distance_upper_bound=thres)
        corresind = corresind[1]
        corresind = corresind[corresind<VT_C.shape[0]]
        corresind[np.sum(np.repeat(NS[i:i+1,:],len(corresind),axis=0)*NT[corresind,:],1)>np.pi] = VT_C.shape[0]
        corresind = corresind[corresind<VT_C.shape[0]]
        len_ = len(corresind)
        #print(corresind)
        if len(corresind)!=0:
            for j in range(len_):
                templength = max(rowlen+1,sum(corres2[corresind[j],:]>0))
                rowlen = sum(corres2[corresind[j],:]>0)
                corres2[corresind[j],rowlen-1] = i
    corres2 = corres2[:, 0:templength]
    reverseStr = []
    rowlen = -1
    for i in range(FT.shape[0]): # find corresponds from target
        corresind = S_tree.query(VT_C[i,:],k=maxind,distance_upper_bound=thres)
        corresind = corresind[1]
        corresind = corresind[corresind<VS_C.shape[0]]
        corresind[np.sum(np.repeat(NT[i:i+1,:],len(corresind),axis=0)*NS[corresind,:],1)>np.pi] = VS_C.shape[0]
        corresind = corresind[corresind<VS_C.shape[0]]
        len_ = len(corresind)
        corres1[i, 0:len_] = corresind
    corres1 = corres1[:, 0:templength]
    tempcorres = np.hstack([corres1,corres2])
    corres = []
    for i in range(tempcorres.shape[0]): # combine and get 0s out
        temp = np.unique(tempcorres[i,:])
        corres.append(temp[temp>0])
    return corres

