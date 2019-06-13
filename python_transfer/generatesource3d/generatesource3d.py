import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import scipy.io
from sklearn.neighbors import NearestNeighbors
from skimage import measure
import matlab
import matlab.engine

def generate_s1_s2(): # generate source 2 from source 1 through tps 3d
    vertices = scipy.io.loadmat('./tmp/vertices.mat')['vertices']
    triangles = scipy.io.loadmat('./tmp/triangles.mat')['triangles']
    obj_file = './tmp/source1.obj'
    with open(obj_file, 'w') as f:
        for v in range(0,vertices.shape[0]):
            f.write('v %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n' % (vertices[v,0],vertices[v,1],vertices[v,2],vertices[v,3],vertices[v,4],vertices[v,5]))
        for t in range(0,triangles.shape[0]):
            f.write('f {} {} {}\n'.format(*triangles[t,:]+1))
    print('Calculated the source 1, save at source1.obj:',obj_file)
    eng = matlab.engine.start_matlab()
    eng.addpath(r'./generatesource3d',nargout=0)
    eng.addpath(r'./tmp',nargout=0)
    eng.Tps_3d(nargout=0)
    vertices = scipy.io.loadmat('./target_vertices.mat')['target_vertices']
    obj_file = './tmp/source2.obj'
    with open(obj_file, 'w') as f:
        for v in range(0,vertices.shape[0]):
            f.write('v %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n' % (vertices[v,0],vertices[v,1],vertices[v,2],vertices[v,3],vertices[v,4],vertices[v,5]))
        for t in range(0,triangles.shape[0]):
            f.write('f {} {} {}\n'.format(*triangles[t,:]+1))
    print('Calculated the source 2, save at source2.obj:',obj_file)
    return vertices

if __name__=='__main__':
    generate_s1_s2()