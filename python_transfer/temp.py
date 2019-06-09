import numpy as np
import non_rigid_registration as non
import normPts as normPts
import build_correspondence as build_c
import scipy.io
import deformation_transfer
data=scipy.io.loadmat('all_you_need.mat')

vs=data['VS']
fs=data['FS']
vt=data['VT']
ft=data['FT']
vs2=data['VS2']
fs2=data['FS2']
marker=data['marker']

all_data = scipy.io.loadmat('./corres')
corres = all_data['corres']

#vs_reg,vt_reg=non.non_rigid_registration(vs,fs,vt,ft,1,0.01,marker)
#print(vs_reg,vt_reg)
#corres=build_c.build_correspondence(vs_reg,fs,vt_reg,ft,10,0.05)
#scipy.io.savemat('./corres_',{'corres':corres})
x,nx=deformation_transfer.deformation_transfer(vs,fs,vt,ft,vs2,fs2,corres)
print('Success')
scipy.io.savemat('vs.mat',{'vertices':x,'triangle':ft})
