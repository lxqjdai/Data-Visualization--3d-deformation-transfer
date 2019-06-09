import numpy.matlib
import numpy as np
def v4_normal(Vert,Face):
    f1=Face[:,0]
    f2=Face[:,1]
    f3=Face[:,2]
    
    e1=Vert[list(f2-1),:]-Vert[list(f1-1),:]
    e2=Vert[list(f3-1),:]-Vert[list(f1-1),:]

    #e2=
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



