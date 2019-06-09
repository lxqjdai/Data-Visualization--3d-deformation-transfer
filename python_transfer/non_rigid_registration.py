import numpy as np
import normPts as normPts
import similarity_fitting as s_f
import v4_normal as v4
import build_adjacency
import build_elementary_cell
import build_phase1 as build1
import python2matlab
import numpy.matlib
def non_rigid_registration(vs,fs,vt,ft,ws,wi,marker):
    
    name=[]
    for i in range(len(marker)):
        name.append(i+1)
        
    S_factor=np.zeros(3)
    T_factor=np.zeros(3)
    S_size=vs.shape[0]

    print("normalize source and target vertices")

    tmean=0
    tstd=2**0.5

    vs=normPts.normPts(vs,tmean,tstd)
    vt=normPts.normPts(vt,tmean,tstd)
    
    print("Align Source to Target vertices")

    R,t,s,res=s_f.similarity_fitting(np.squeeze(vt[marker[:,1:2]]),np.squeeze(vs[marker[:,0:1]]))

    vt=np.matmul(vt,np.transpose(s*R))+np.matlib.repmat(t,len(vt),1)

    ts,ns,vs4,fs4=v4.v4_normal(vs,fs)
    tt,nt,vt4,ft4=v4.v4_normal(vt,ft)

    print("Building adjacency...")
    Adj_idx=build_adjacency.build_adjacency(fs)

    print("Solving Phase 1 optimization...")
    E=build_elementary_cell.build_elementary_cell(ts,len(fs))

    I1,I2,I3,C = build1.build_phase1(Adj_idx,E,fs4,vt4,ws,wi,marker)

    print('Build phase 1 finished')

    VSP1=np.array(python2matlab.matlabsolver())

    VSP1=np.reshape(VSP1,[3,int(VSP1.shape[0]/3)],order='F')
    VSP1=np.transpose(VSP1)
    VSP1=VSP1[0:S_size,:]
    print('VSP-finished')
    return VSP1,vt

    


