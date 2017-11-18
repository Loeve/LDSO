import numpy as np
import math
# import multiprocessing as mp
import bp
# import BP_salient_map

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

def mkNodeFeatures(raw):
    draw,nr, nc, ncases=raw.shape
    D=draw+1
    nodeFeatures = np.zeros((D, nr, nc, ncases))
    for s in range(0,ncases):
        for r in range(0,nr):
            for c in range(0,nc):
                nodeFeatures[0,r,c,s]=1
                nodeFeatures[1:, r, c, s] =  raw[:, r, c, s]
    nodeFeatures=np.reshape(nodeFeatures,(D,nr*nc,ncases))
    return nodeFeatures

def getMu( vec1, vec2):
    n=vec1.shape[0]+1
    v1=np.zeros((n))
    v2=np.zeros((n))
    mu=np.zeros((n))
    v1[0]=1
    v1[1:]=vec1
    v2[0]=1
    v2[1:]=vec2
    for i in range(0,n):
        mu[i]=math.fabs(v1[i]-v2[i])
    norm_mu=np.linalg.norm(mu,ord=2)
    if(norm_mu!=0):
        mu=mu/norm_mu
    mu[0]=1
    return mu

def legal(i,j,nrows,ncols):
    if (i >= 0 and j >= 0 and i < nrows and j < ncols):
        bool = 1
    else:
        bool = 0
    return bool

def subv2ind(siz, subv):
    # ncases, ndims=subv.shape
    n=siz.shape[0]
    siztemp=siz[0:n-1].cumprod()
    cp=np.zeros((siztemp.shape[0]+1))
    cp[0]=1
    cp[1:]=siztemp
    ndx = np.dot(subv , cp) + 1   #different
    return ndx

def assign_edge_nums_lattice(nrows, ncols):
    north = 0
    east = 1
    south = 2
    west = 3
    ndir = 4
    out_edge = np.zeros((nrows, ncols, ndir))
    in_edge = np.zeros((nrows, ncols, ndir))
    outNbr = np.zeros((nrows, ncols, ndir, 2))
    nedgesUndir = nrows * (ncols - 1) + ncols * (nrows - 1)
    nedgesDir = nedgesUndir * 2
    edgeEndsRC = np.zeros((nedgesDir, 4))
    edgeEndsIJ = np.zeros((nedgesDir, 2))
    e = 0
    for c in range(0,ncols):
        for r in range(0,nrows):
            if legal(r,c-1,nrows,ncols):
                out_edge[r,c,west] = e
                in_edge[r,c-1,east] = e
                outNbr[r,c,west,:] = [r, c-1]
                edgeEndsRC[e,:] = [r,c ,r, c-1]
                e = e+1
            if legal(r-1,c,nrows,ncols):
                out_edge[r,c,north] = e
                in_edge[r-1,c,south] = e
                outNbr[r,c,north,:] = [r-1, c]
                edgeEndsRC[e,:] = [r,c ,r-1, c]
                e = e+1
            if legal(r+1,c,nrows,ncols):
                out_edge[r,c,south] = e
                in_edge[r+1,c,north] = e
                outNbr[r,c,south,:] = [r+1, c]
                edgeEndsRC[e,:] = [r,c ,r+1, c]
                e = e+1
            if legal(r,c+1,nrows,ncols):
                out_edge[r,c,east] = e
                in_edge[r,c+1,west] = e
                outNbr[r,c,east,:] = [r, c+1]
                edgeEndsRC[e,:] = [r,c ,r, c+1]
                e = e+1
    nedges = e
    temp=np.array([nrows, ncols])
    edgeEndsIJ[:, 0] = subv2ind(temp, edgeEndsRC[:, 0: 2])
    edgeEndsIJ[:, 1] = subv2ind(temp, edgeEndsRC[:, 2: 4])
    return  out_edge, outNbr, edgeEndsIJ, edgeEndsRC, in_edge, nedges

def mkEdgeFeatures(featureEng, raw):
    Draw,nr,nc,ncases=raw.shape
    expand =featureEng.expandEdge
    vec1=raw[:,0,0,0]
    vec2=raw[:,0,0,0]
    D = len(getMu(vec1,vec2))
    out_edge, outNbr, edgeEndsIJ, edgeEndsRC, in_edge, nedges=assign_edge_nums_lattice(nr, nc)
    nedgesDir=edgeEndsIJ.shape[0]
    edgeFeatures=np.zeros(((D, nedgesDir, ncases)))
    for s in range(0,ncases):
        for e in range(0,nedgesDir):
            r1 = int(edgeEndsRC[e, 0])
            c1 = int(edgeEndsRC[e, 1])
            r2 = int(edgeEndsRC[e, 2])
            c2 = int(edgeEndsRC[e, 3])
            mu = getMu(raw[:, r1, c1, s], raw[:, r2, c2, s])
            edgeFeatures[:, e, s] = mu
    return edgeFeatures

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
    featureEng.edgeDirNum, featureEng.outNbr, featureEng.edgeEndsIJ, featureEng.edgeEndsRC,in_edge, nedges=assign_edge_nums_lattice(nr,nc)
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
            if legal(i,j-1,nr,nc):
                E.append(node-nr)
                edge+=1
            if legal(i-1,j,nr,nc):
                E.append(node-1)
                edge+=1
            if legal(i+1,j,nr,nc):
                E.append(node+1)
                edge+=1
            if legal(i,j+1,nr,nc):
                E.append(node+nr)
                edge+=1
            node+=1
    V.append(edge)
    nedges=edge-1
    return V,E,nedges

# def job(x):
#     return x-1

def infer(infEng, nodePot, edgePot):
    nr,nc,nstates=nodePot.shape
    # nNodes=nr*nc
    nodePot=(np.reshape(nodePot,(nr*nc,nstates))).astype(float)
    edgePot=edgePot.astype(float)
    v, e, nEdges = StarEdge_MakeEdgeNums_Lattice2(nr, nc)
    v1=np.array(v)
    e1=np.array(e)
    # # BP_infer(nodePot0,edgePot0,starV,starE,maximize,maxIter,optTol)
    # nodeBel=BP_salient_map.BP_infer(nodePot,edgePot,infEng,v1,e1)
    new_bel = bp.BP_infer(nodePot, edgePot, v1, e1, infEng.maximize, infEng.maxIter, infEng.tol)
    nodeBel= np.reshape(new_bel, (nr,nc,nstates))
    mrc=np.zeros((nr,nc))
    for i in range(0,nr):
        for j in range(0,nc):
            if nodeBel[i,j,0]>nodeBel[i,j,1]:
                mrc[i,j]=1
            else:
                mrc[i,j]=0
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
    td.nodeFeatures=mkNodeFeatures(testFeatures)
    td.edgeFeatures=mkEdgeFeatures(featureEng,testFeatures)
    labeled_masks_test=np.zeros((height,width,1))
    td.nodeLabels=labeled_masks_test
    td.ncases=1
    weights = np.array([-0.1245, 0.5379, 0.7741, 0.7778, 0.8486, 0.2229, 0.3007, 0.3843])
    featureEng=enterEvidence(featureEng,td,0)
    nodePot,edgePot=mkPotentials(featureEng,weights)
    infEng=latticeInferBP(nstates)
    ######################################
    MAPlabels=infer(infEng, nodePot, edgePot)
    # salient_map=(MAPlabels+1)*0.5
    return MAPlabels


