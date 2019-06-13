import cv2 as cv
import dlib
import numpy
import sys
import matplotlib.pyplot as plt
import math

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    # the feature extractor (predictor) requires a rough bounding box as input
    # to the algorithm. This is provided by a traditional face detector (
    # detector) which returns a list of rectangles, each of which corresponding
    # a face in the image
    return numpy.array([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

for i in range(1,41):
    image_file='data/'+str(i)+'.jpg'
    image = cv.imread(image_file)
    temp=get_landmarks(image)
    preds=temp
    image = cv.imread(image_file)
    try:
        image_height, image_width, image_depth = image.shape
    except:
        print('cannot load image:', image_file)
    minX=1000
    maxX=0
    minY=1000
    maxY=0
    for var in  preds:
        if minX > var[0]:
            minX = var[0]
        if maxX < var[0]:
            maxX = var[0]
        if minY > var[1]:
            minY = var[1]
        if maxY < var[1]:
            maxY = var[1]
   
    ### crop face image
    scale=90/math.sqrt((minX-maxX)*(minY-maxY))
    width=maxX-minX
    height=maxY-minY
    cenX=width/2
    cenY=height/2

    x= int( (minX+cenX)*scale )
    y= int( (minY+cenY)*scale )
#print x,y,scale

    resized_image = cv.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    rh,rw,rc =  resized_image.shape

    #
    crop_width = 160
    crop_height = 244
    left = 0
    top = 0
    right = 0
    bottom = 0
    cx = x
    cy = y
    
    if x < crop_width/2:
        left = int(crop_width/2 - x)
        cx = x + left
    if y < crop_height/2:
        top = crop_height/2 - y
        cy = y + top
    if rw - x < crop_width/2:
        right =  crop_width/2 + x - rw;
    if rh - y < crop_height/2:
        bottom = crop_height/2 + y - rh
    #
    
    crop_image = cv.copyMakeBorder(resized_image,top, int(bottom), int(left), int(right),cv.BORDER_REFLECT)

    crop_image = crop_image[cy-int(crop_height//2):cy+int(crop_height/2), cx-int(crop_width/2):cx+int(crop_width/2), :]
    cv.imwrite(str(i)+'.jpg',crop_image)
