import numpy as np
import non_rigid_registration as non
import normPts as normPts
import build_correspondence as build_c
import scipy.io
import deformation_transfer
import python2matlab
import v4_normal as v4
from create.create import create_boban, create_3d_marker
from generatesource3d.generatesource3d import generate_s1_s2
#data=scipy.io.loadmat('all_you_need.mat')

vs_data,fs,marker1=create_boban('./source1.jpg','./source2.jpg')
vs2_data,fs2 = generate_s1_s2(),fs
vt_data,ft,marker2 = create_3d_marker('./target1.jpg')
marker = np.transpose(np.vstack((marker1,np.transpose(marker2))))
vs,vs2,vt = vs_data[:,0:3], vs2_data[:,0:3],vt_data[:,0:3]
'''
vs=data['VS']
fs=data['FS']
vt=data['VT']
ft=data['FT']
vs2=data['VS2']
fs2=data['FS2']
marker=data['marker']
'''
vs_reg,vt_reg=non.non_rigid_registration(vs,fs,vt,ft,1,0.1,marker)
#print(vs_reg,vt_reg)
corres=build_c.build_correspondence(vs_reg,fs,vt_reg,ft,10,0.05)
#scipy.io.savemat('./corres.mat',{'corres':corres})
x,nx=deformation_transfer.deformation_transfer(vs,fs,vt,ft,vs2,fs2,corres)

print('Success')
scipy.io.savemat('vs_2.mat',{'vertices':x,'triangle':ft,'n_vertices':nx})
