import numpy as np
import math
import bp
import Infer_map
from skimage import  segmentation, measure, morphology,filters
import matplotlib.patches as mpatches

class latticeFeatures:
    nstates = 2
    nnodes = []
    nedges = []
    nrows = []
    ncols = []
    Dnode = []
    Dedge = []
    nodeFeatures = []
    edgeFeatures = []
    nodeLabels = []
    edgeDirNum = []
    outNbr = []
    edgeEndsIJ = []
    edgeEndsRC = []
    expandNode= 0
    expandEdge= 0

class testdata:
    nodeFeatures=[]
    edgeFeatures=[]
    nodeLabels=[]
    ncases=[]

class latticeInferBP:
    nstates = []
    maxIter = []
    tol = []
    maximize = []

def mkPotentials(featureEng, weights):
    nodeWeights=weights[0:featureEng.Dnode]
    edgeWeights=weights[featureEng.Dnode:]
    nodetmp=np.dot(nodeWeights.T,featureEng.nodeFeatures)
    nexptmp1=np.zeros(nodetmp.shape)
    nexptmp2=np.zeros(nodetmp.shape)
    for k in range(0,nodetmp.shape[0]):
        nexptmp1[k]=math.exp(nodetmp[k])
        nexptmp2[k]=math.exp(-nodetmp[k])
    nodePot=np.row_stack([nexptmp1,nexptmp2])
    nodePot=np.reshape(nodePot.T,(featureEng.nrows, featureEng.ncols, featureEng.nstates))
    edgetmp=np.dot(edgeWeights.T,featureEng.edgeFeatures)
    edgePot=np.zeros((2,2,edgetmp.shape[0]))
    dexptmp1 = np.zeros(edgetmp.shape)
    dexptmp2 = np.zeros(edgetmp.shape)
    for k in range(0,edgetmp.shape[0]):
        dexptmp1[k]=math.exp(edgetmp[k])
        dexptmp2[k]=math.exp(-edgetmp[k])
    edgePot[0,0,:] = dexptmp1
    edgePot[0,1,:] = dexptmp2
    edgePot[1,0,:] =dexptmp2
    edgePot[1,1,:] = dexptmp1
    return nodePot, edgePot

def latticeInferBP(nstates):
    # firstArg = nstates
    obj=latticeInferBP
    obj.nstates=nstates
    obj.maxIter=200
    obj.tol=1e-3
    obj.maximize=0
    return obj

def enterEvidence(featureEng, data, s):
    nr,nc,ncases=data.nodeLabels.shape
    featureEng.nodeLabels=np.reshape(data.nodeLabels[:,:,s],(nr*nc,1))
    featureEng.nodeFeatures=data.nodeFeatures[:,:,s]
    featureEng.edgeFeatures=data.edgeFeatures[:,:,s]
    featureEng.edgeDirNum, featureEng.outNbr, featureEng.edgeEndsIJ, featureEng.edgeEndsRC,in_edge, nedges=Infer_map.assign_edge_nums_lattice(nr,nc)
    featureEng.nrows=nr
    featureEng.ncols=nc
    featureEng.nnodes=nr*nc
    featureEng.Dnode=data.nodeFeatures.shape[0]
    featureEng.Dedge=data.edgeFeatures.shape[0]
    return featureEng

def StarEdge_MakeEdgeNums_Lattice2(nr,nc):
    node=1
    edge=1
    V=[]
    E=[]
    for j in range(0,nc):
        for i in range(0,nr):
            V.append(edge)
            if Infer_map.legal(i,j-1,nr,nc):
                E.append(node-nr)
                edge+=1
            if Infer_map.legal(i-1,j,nr,nc):
                E.append(node-1)
                edge+=1
            if Infer_map.legal(i+1,j,nr,nc):
                E.append(node+1)
                edge+=1
            if Infer_map.legal(i,j+1,nr,nc):
                E.append(node+nr)
                edge+=1
            node+=1
    V.append(edge)
    nedges=edge-1
    return V,E,nedges

def infer(infEng, nodePot, edgePot):
    nr,nc,nstates=nodePot.shape
    nodePot=(np.reshape(nodePot,(nr*nc,nstates))).astype(float)
    edgePot=edgePot.astype(float)
    v, e, nEdges = StarEdge_MakeEdgeNums_Lattice2(nr, nc)
    v1=np.array(v)
    e1=np.array(e)
    mrc = bp.BP_infer(nodePot, edgePot, v1, e1, infEng.maximize, infEng.maxIter, infEng.tol,nr,nc)

    return mrc

def infer_salient_map(contrast_map,center_surround_map,color_spatial_map):
    height, width=contrast_map.shape
    contrast_map = 10.0 * contrast_map - 2.0
    center_surround_map = 8.0 * center_surround_map - 3.0
    color_spatial_map = 8.0 * color_spatial_map - 3.0
    feature_maps = np.zeros((3, height, width, 1))
    feature_maps[0,:,:, 0] = contrast_map
    feature_maps[1,:,:, 0] = center_surround_map
    feature_maps[2,:,:, 0] = color_spatial_map
    featureEng=latticeFeatures
    nstates=2
    testFeatures = feature_maps
    td=testdata
    td.nodeFeatures=Infer_map.mkNodeFeatures(testFeatures)
    td.edgeFeatures=Infer_map.mkEdgeFeatures(featureEng,testFeatures)
    labeled_masks_test=np.zeros((height,width,1))
    td.nodeLabels=labeled_masks_test
    td.ncases=1
    weights = np.array([-0.1245, 0.5379, 0.7741, 0.7778, 0.8486, 0.2229, 0.3007, 0.3843])
    featureEng=enterEvidence(featureEng,td,0)
    nodePot,edgePot=mkPotentials(featureEng,weights)
    infEng=latticeInferBP(nstates)
    MAPlabels=infer(infEng, nodePot, edgePot)

    return MAPlabels

def draw_rectangle(salient_map):
    thresh = filters.threshold_otsu(salient_map)
    bw = morphology.closing(salient_map > thresh, morphology.square(3))
    cleared = bw.copy()
    segmentation.clear_border(cleared)
    label_image = measure.label(cleared)
    arealist = []
    for region in measure.regionprops(label_image):
        arealist.append(region.area)
    index = arealist.index(max(arealist))  # find the index of the max(arealist)
    minr, minc, maxr, maxc = measure.regionprops(label_image)[index].bbox
    rect1 = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1)
    rect2 = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='green', linewidth=1)

    return rect1,rect2