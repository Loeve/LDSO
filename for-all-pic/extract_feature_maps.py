import numpy as np
import cv2
import math
from sklearn.cluster import KMeans
import EFM

def mult_con_map(img):
    img=img.astype(float)
    level=6
    h,w,c=img.shape
    multmap=np.zeros((h,w))
    dx = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    dy = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    for i in range(level):
        fimg=cv2.GaussianBlur(img,(5,5),1.5)
        w1=int(w/pow(2,i))
        h1=int(h/pow(2,i))
        scalimage=cv2.resize(fimg,(w1,h1),cv2.INTER_LINEAR)
        scalmap=np.zeros((h1,w1))
        for row in range(1,h1-1):
            for col in range(1,w1-1):
                contrast_value=0
                distance=0
                for window in range(0,9):
                    intensity=scalimage[row,col,:]
                    neibor_intersity=scalimage[row+dy[window],col+dx[window],:]
                    distance=distance+sum(pow((intensity-neibor_intersity),2))
                    contrast_value+=distance
                scalmap[row,col]=contrast_value
        scalmap=(scalmap-np.min(scalmap))/(np.max(scalmap)-np.min(scalmap))
        rescalmap=cv2.resize(scalmap,(w,h),cv2.INTER_LINEAR)
        multmap=multmap+rescalmap
        img=fimg
    multmap=(multmap-np.min(multmap))/(np.max(multmap)-np.min(multmap))
    return multmap

def EM_init_kmeans(Data, nbStates):
    nbVar, nbData = Data.shape
    estimator=KMeans(nbStates)
    estimator.fit(Data.T)
    Data_id=estimator.labels_
    Centers=estimator.cluster_centers_
    mu=Centers.T
    priors=np.zeros((1,nbStates))
    sigma=np.zeros((nbVar,nbVar,nbStates))
    for i in range(0,nbStates):
        idtemp=np.where(Data_id==i)
        priors[0,i]=len(idtemp[0])
        covtemp=np.column_stack([Data[:,idtemp[0]],Data[:,idtemp[0]]])
        sigma[:,:,i]=np.cov(covtemp)
        sigma[:,:,i]=sigma[:,:,i]+1e-5*np.eye((nbVar))
    priors=priors/priors.sum()

    return priors,mu,sigma

def gaussPDF(Data, Mu, Sigma):
    nbVar, nbData=Data.shape
    Data=Data.T-np.tile(Mu.T,(nbData,1))
    prob=((np.dot(Data,np.linalg.inv(Sigma)))*Data).sum(axis=1)
    for i in range(0,nbData):
        try:
            prob[i]=math.exp(-0.5*prob[i])
        except OverflowError:
            prob[i]=float('inf')
    prob=prob/math.sqrt(math.pow((2*math.pi),nbVar)*(math.fabs(np.linalg.det(Sigma))+2.2251e-308))
    return prob

def EM(data, priors0, mu0, sigma0):
    loglike_threshold=1e-3
    nbVar, nbData=data.shape
    nbStates=sigma0.shape[2]
    loglik_old=-(1.7977e+308)
    nbStep=0
    mu=mu0
    sigma=sigma0
    priors=priors0
    pix=np.zeros((nbData,nbStates))

    while 1:

        for i in range(0,nbStates):
            pix[:,i]=gaussPDF(data, mu[:,i], sigma[:,:,i])
        pix_tmp=np.tile(priors,(nbData,1))*pix
        temper=np.tile(pix_tmp.sum(axis=1),(nbStates,1))
        pix=pix_tmp/(temper.T)     ##################
        E=sum(pix)
        for i in range(0,nbStates):
            priors[0,i]=E[i]/nbData
            mu[:,i]=np.dot(data,pix[:,i])/E[i]
            data_tmp1=data-(np.tile(mu[:,i],(nbData,1))).T
            sigma[:,:,i]=np.dot(np.tile(pix[:,i].T,(nbVar,1))*data_tmp1,(data_tmp1.T))/E[i]
            sigma[:,:,i]=sigma[:,:,i]+1e-5*np.diag(np.ones((nbVar,1))[0])
        for i in range(0,nbStates):
            sigma[:, :, i] = sigma[:, :, i] + 1e-5 * np.eye((nbVar))
            pix[:,i]=gaussPDF(data,mu[:,i],sigma[:,:,i])
        F=np.dot(pix,priors.T)
        F[np.where(F<2.2251e-308)]=2.2251e-308       ##############
        loglik=np.mean(np.log(F))
        if np.fabs((loglik/loglik_old)-1)<loglike_threshold:     ##############
            break
        loglik_old=loglik
        nbStep=nbStep+1

    return priors,mu,sigma

def generate_colormaps(image, numberOfColors):
    height, width, c = image.shape
    pixelValues=np.zeros((3,height*width))
    pixelValues[0] = np.reshape(image[:, :, 0], (1,height*width))
    pixelValues[1] = np.reshape(image[:, :, 1], (1,height*width))
    pixelValues[2] = np.reshape(image[:, :, 2], (1,height*width))
    pixelValues=pixelValues.astype(float)
    pixelValues=pixelValues/np.max(pixelValues)
    Weights0, Mu0, Sigma0= EM_init_kmeans(pixelValues, numberOfColors)
    Weights, Mu, Sigma = EM(pixelValues, Weights0, Mu0, Sigma0)
    colormaps=np.zeros((height,width,numberOfColors))
    for i in range(0,numberOfColors):
        probabilities = gaussPDF(pixelValues, Mu[:,i], Sigma[:,:, i])
        colormaps[:,:, i] = np.reshape(probabilities,(height, width))
    cumulated_colormap = np.zeros((height, width))
    for i in range(0,numberOfColors):
        cumulated_colormap=cumulated_colormap+Weights[0,i]*colormaps[:,:,i]
    for i in range(0,numberOfColors):
        colormaps[:,:,i]=Weights[0,i]*colormaps[:,:,i]/cumulated_colormap
    return colormaps

def calculate_spatial_variances(colormaps):
    horizontal_variances = EFM.calculate_horizontal_variances(colormaps)
    vertical_variances = EFM.calculate_vertical_variances(colormaps)
    variances = horizontal_variances + vertical_variances
    variances = (variances - np.min(variances[:]))/ (np.max(variances[:]) - np.min(variances[:]))
    return variances


