import cv2
import dlib
import numpy as np
import scipy
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import create.face_alignment as face_alignment
import create.vrn_unguided as vrn_unguided
from skimage import measure
from sklearn.neighbors import NearestNeighbors

PREDICTOR_PATH = "./create/shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATURE_AMOUNT = 11

#68个点属于不同的分类
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
BROW_POINTS=list(range(17,27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS
                  + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
    ]

COLOUR_CORRECT_BLUR_FRAC = 0.05
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

VRN = vrn_unguided.vrn_unguided
VRN.load_state_dict(torch.load('./create/models/vrn_unguided.pth'))
enable_cuda=False

def get_landmarks(im):
    img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)   
    # 人脸数rects
    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        land = np.matrix([[p.x, p.y] for p in predictor(im,rects[i]).parts()])
    return land

#帮助讲关键点画在脸部进行展示
def annote_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def transformation_from_points(points1, points2):   
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])
#获得对齐后的图片
def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def create_boban(source1,source2):
    '''
    bobanyangtiao source1,source2 and marker
    '''
    source1 = cv2.imread(source1)
    source2 = cv2.imread(source2)
    landmarks1=get_landmarks(source1)
    landmarks2=get_landmarks(source2)
    M = transformation_from_points(landmarks1,
                                landmarks2)
    source2=warp_im(source2,M,source1.shape)
    landmarks2=get_landmarks(source2)

    inp = torch.from_numpy(source1.transpose((2, 0, 1))).float().unsqueeze_(0)
    if enable_cuda:
        inp = inp.cuda()
    out = VRN(Variable(inp, volatile=True))[-1].data.cpu()
    im =  source1[:,:,[2,1,0]] #RGB
    vol = out.numpy()
    vol = vol.reshape((200,192,192))*255.0
    vol = vol.astype(float)
    a = measure.marching_cubes_lewiner(vol, 10)
    vertices=a[0]
    triangles=a[1]
    vertices = vertices[:,(2,1,0)]
    vertices[:,2] *= 0.5 # scale the Z component correctly
    r = im[:,:,0].flatten()
    g = im[:,:,1].flatten()
    b = im[:,:,2].flatten()

    vcx,vcy = np.meshgrid(np.arange(0,192),np.arange(0,192))
    vcx = vcx.flatten()
    vcy = vcy.flatten()
    vc = np.vstack((vcx, vcy, r, g, b)).transpose()
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(vc[:,:2])
    n = neigh.kneighbors(vertices[:,(0,1)], return_distance=False)
    colour = vc[n,2:].reshape((vertices.shape[0],3)).astype(float) / 255

    vc = np.hstack((vertices, colour))

    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(vertices[:,:2])
    nn=neigh.kneighbors(landmarks1,return_distance=False)
    nnn=[]
    for i in range(68):
        a=vc[nn[:,0]][i][2]
        b=vc[nn[:,1]][i][2]
        if a>b:
            nnn.append(nn[:,0][i])
        else:
            nnn.append(nn[:,1][i])
    original_point=vc[nnn][:,:3]
    target_point=np.hstack((landmarks2,vc[nnn][:,2:3]))
    scipy.io.savemat('./tmp/vertices.mat',{'vertices':vc})
    scipy.io.savemat('./tmp/triangles.mat',{'triangles':triangles})
    scipy.io.savemat('./tmp/original_point.mat',{'original_point':original_point})
    scipy.io.savemat('./tmp/target_point.mat',{'target_point':target_point})
    return vc,triangles,nnn

def create_3d_marker(target):
    '''
    create 3d and marker
    '''
    target = cv2.imread(target)
    landmarks=get_landmarks(target)
    inp = torch.from_numpy(target.transpose((2, 0, 1))).float().unsqueeze_(0)
    if enable_cuda:
        inp = inp.cuda()
    out = VRN(Variable(inp, volatile=True))[-1].data.cpu()
    ### save to obj file

    im =  target[:,:,[2,1,0]] #RGB
    vol = out.numpy()
    vol = vol.reshape((200,192,192))*255.0
    vol = vol.astype(float)

    a = measure.marching_cubes_lewiner(vol, 10)
    vertices=a[0]
    triangles=a[1]
    vertices = vertices[:,(2,1,0)]
    vertices[:,2] *= 0.5 # scale the Z component correctly

    r = im[:,:,0].flatten()
    g = im[:,:,1].flatten()
    b = im[:,:,2].flatten()

    vcx,vcy = np.meshgrid(np.arange(0,192),np.arange(0,192))
    vcx = vcx.flatten()
    vcy = vcy.flatten()
    vc = np.vstack((vcx, vcy, r, g, b)).transpose()
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(vc[:,:2])
    n = neigh.kneighbors(vertices[:,(0,1)], return_distance=False)
    colour = vc[n,2:].reshape((vertices.shape[0],3)).astype(float) / 255

    vc = np.hstack((vertices, colour))
    '''
    obj_file = 'obama.obj'
    with open(obj_file, 'w') as f:
        for v in range(0,vc.shape[0]):
            f.write('v %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n' % (vc[v,0],vc[v,1],vc[v,2],vc[v,3],vc[v,4],vc[v,5]))

        for t in range(0,triangles.shape[0]):
            f.write('f {} {} {}\n'.format(*triangles[t,:]+1))

    print('Calculated the isosurface, save at obj file:',obj_file)
    '''
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(vertices[:,:2])
    nn=neigh.kneighbors(landmarks,return_distance=False)
    nnn=[]
    for i in range(68):
        a=vc[nn[:,0]][i][2]
        b=vc[nn[:,1]][i][2]
        if a>b:
            nnn.append(nn[:,0][i])
        else:
            nnn.append(nn[:,1][i])
    return vc,triangles,nnn

#source1=cv2.imread('source1.jpg')
#source2=cv2.imread("dazuichun.jpg")
#target=cv2.imread('target1.jpg')




