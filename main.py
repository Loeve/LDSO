import numpy as np
from extract_feature_maps import *
from infer_salient_map import *
import cv2
import time
import EFM  #cython:6s ; python:22s


start=time.clock()
image = cv2.imread('testt.jpg',cv2.IMREAD_COLOR)
h,w,c=image.shape
nh=int(200*h/max(w,h))
nw=int(200*w/max(w,h))
#######与MATLAB双线性插值后结果不一样###########
img=cv2.resize(image,(nw,nh),cv2.INTER_LINEAR)

# img = cv2.imread('101.jpg',cv2.IMREAD_COLOR)
img=img.astype(float)

contrast_map=mult_con_map(img)
map,rectangle=EFM.distance_map(img)
center_surround_map = EFM.center_surround_histogram_map(map,rectangle)
colormaps=generate_colormaps(img,6)
variances=calculate_spatial_variances(colormaps)
color_spatial_map = EFM.cw_csd_map(colormaps,variances)

# contrast_map=np.zeros((400,300))

salient_map= infer_salient_map(contrast_map,center_surround_map,color_spatial_map)
cv2.imshow("salient_map",salient_map)
# # # cv2.imshow('image',image)
# # cv2.imshow('img',img)
# cv2.imshow('contrast_map',contrast_map)
# cv2.imshow('center_surround_map',center_surround_map)
# cv2.imshow('color_spatial_map',color_spatial_map)
end=time.clock()
print('runing time: %s seconds'%(end-start))
cv2.waitKey(0)




