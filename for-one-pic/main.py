from extract_feature_maps import *
from infer_salient_map import *
import EFM  #cython:6s ; python:22s
import cv2
import time
from skimage import io
import matplotlib.pyplot as plt
import os


test_pic_dir="test_images"
images=os.listdir(test_pic_dir)    #file names of all images which need to be computed(eg:'101.jpg')
ground_truth_dir="ground_turth"
truths=os.listdir(ground_truth_dir)   #file names all ground truth images(eg:'101.png')
resutl_map_dir="result_map"
k=0
for i in images:
    start = time.clock()
    k+=1
    name = i[:i.index(".")]  # name of image(eg:101)
    image=io.imread(test_pic_dir+"/"+i)
    h,w,c=image.shape
    nh=int(200*h/max(w,h))
    nw=int(200*w/max(w,h))
    img=cv2.resize(image,(nw,nh),cv2.INTER_LINEAR)
    ######### compute feature maps
    contrast_map=mult_con_map(img.astype(float))
    map,rectangle=EFM.distance_map(img.astype(float))
    center_surround_map = EFM.center_surround_histogram_map(map,rectangle)
    colormaps=generate_colormaps(img.astype(float),6)
    variances=calculate_spatial_variances(colormaps)
    color_spatial_map = EFM.cw_csd_map(colormaps,variances)
    ####### compute salient map
    salient_map= infer_salient_map(contrast_map,center_surround_map,color_spatial_map)
    end = time.clock()
    print('computing time for pic %s: %s seconds' % (k, (end - start)))
    ########## show and save result
    rect1,rect2=draw_rectangle(salient_map)
    fig=plt.figure()
    plt.subplot(231),plt.title("source image"),plt.imshow(img)
    plt.subplot(231).add_patch(rect1)
    plt.subplot(232),plt.title("salient_map"),plt.imshow(salient_map,plt.cm.gray)
    plt.subplot(232).add_patch(rect2)
    ################################   show ground
    if((name+".png") in truths):
        ground_truth=io.imread(ground_truth_dir+"/"+name+".png")
        plt.subplot(233),plt.title("ground_truth"),plt.imshow(ground_truth,plt.cm.gray)
    ########## show feature maps
    plt.subplot(234),plt.title("contrast_map"),plt.imshow(contrast_map,plt.cm.gray)
    plt.subplot(235),plt.title("center_surround_map"),plt.imshow(center_surround_map,plt.cm.gray)
    plt.subplot(236),plt.title("color_spatial_map"),plt.imshow(color_spatial_map,plt.cm.gray)
    plt.savefig(resutl_map_dir+"/"+name)
    # plt.show()
    # plt.pause(1)
    # plt.close()





