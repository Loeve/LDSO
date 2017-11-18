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

# def rgb_labeled_map(img):
#     red=img[:,:,0]/255*99/10
#     green=img[:,:,1]/255*99/10
#     blue=img[:,:,2]/255*99/10
#     red = red.astype(int)
#     green = green.astype(int)
#     blue = blue.astype(int)
#     label_map=100*blue+10*green+red
#     return label_map

# def integral_histogram(map):
#     h,w=map.shape
#     inte_hist=np.zeros((h,w,1000))
#     inte_hist[0,0,map[0,0]]+=1
#     for k in range(1,w):
#         inte_hist[0,k]+=inte_hist[0,k-1]
#         inte_hist[0,k,map[0,k]]+=1
#     for k in range(1,h):
#         inte_hist[k,0]+=inte_hist[k-1,0]
#         inte_hist[k,0,map[k,0]]+=1
#     for i in range(1,h):
#         for j in range(1,w):
#             inte_hist[i,j]=inte_hist[i-1,j]+inte_hist[i,j-1]-inte_hist[i-1,j-1]
#             inte_hist[i,j,map[i,j]]+=1
#     return inte_hist

# def generate_rect(x,y,scale,ratio):
#     if ratio>1.0:
#         h=scale
#         w=math.floor(h/ratio)
#     else:
#         w=scale
#         h=math.floor(w*ratio)
#     left=y-math.floor(w/2)
#     top=x-math.floor(h/2)
#     rect=np.array([left,top,w,h])
#     return rect

# def calculate_surrounding_rectangle(inner_rectangle):
#     left = inner_rectangle[0]
#     top = inner_rectangle[1]
#     width = inner_rectangle[2]
#     height = inner_rectangle[3]
#     surrounding_width = math.floor(pow(2,0.5) * width)
#     surrounding_height = math.floor(pow(2,0.5) * height)
#     surrounding_left = left - math.floor((pow(2,0.5) - 1) * width / 2)
#     surrounding_top = top - math.floor((pow(2,0.5) - 1) * height / 2)
#     surrounding_rectangle =np.array([surrounding_left,surrounding_top,surrounding_width,surrounding_height])
#     return surrounding_rectangle

# def check_that_rectangle_is_inside(RS,w,h):
#     rleft = RS[0]
#     rtop = RS[1]
#     rwidth = RS[2]
#     rheight = RS[3]
#     result = 1
#     if (rleft < 0):
#         result = 0
#     if (rtop < 0):
#         result = 0
#     if (rleft + rwidth >= w):
#         result = 0
#     if (rtop + rheight >= h):
#         result = 0
#     return result

# def encompass(rectangleA, rectangleB):
#     left = rectangleA[0]
#     top = rectangleA[1]
#     width = rectangleA[2]
#     height = rectangleA[3]
#     if (left < rectangleB[0]):
#         left = rectangleB[0]
#     if (top < rectangleB[1]):
#         top = rectangleB[1]
#     if (left + width > rectangleB[0] + rectangleB[2]):
#         width = rectangleB[0] + rectangleB[2] - left - 1
#     if (top + height > rectangleB[1] + rectangleB[3]):
#         height = rectangleB[1] + rectangleB[3] - top - 1
#     return left,top,width,height

# def calculate_histogram_fast(labeled_map,inte_hist,rectangle):
#     h,w=labeled_map.shape
#     image_rectangle = np.array([1,1,w,h])
#     [left, top, width, height] = encompass(rectangle, image_rectangle)
#     iMin = top
#     iMax = top + height - 1
#     jMin = left
#     jMax = left + width - 1
#     histogram=inte_hist[iMax,jMax]+inte_hist[iMin,jMin]-inte_hist[iMin,jMax]-inte_hist[iMax,jMin]
#     return histogram

# def chi_square_distance(vectorA, vectorB):
#     # vectorA = float(vectorA)
#     # vectorB = float(vectorB)
#     if sum(vectorA):
#         vectorA = vectorA / sum(vectorA)
#     if sum(vectorB):
#         vectorB = vectorB / sum(vectorB)
#     distance = 0.0
#     squared_difference = pow((vectorA - vectorB), 2)
#     sum_vector = vectorA + vectorB
#     lengthA = vectorA.shape[0]
#     for i in range(1,lengthA):
#         denominator = sum_vector[i]
#         if (denominator > 0):
#             distance = distance + (squared_difference[i] / denominator)
#     distance = 0.5 * distance
#     return distance

# def saliency_at_position(labeled_map,inte_hist,r,c):
#     h,w=labeled_map.shape
#     initial=int(0.1*min(h,w))
#     final=int(0.7*min(h,w))
#     lstep=int(0.1*min(h,w))
#     best_rect=np.array([0,0,0,0])
#     aspect_ratios=np.array([0.5,0.75,1.0,1.5,2.0])
#     max_chi_square_distance=0
#     for scale in range(initial,final,lstep):
#         for index in range(0,5):
#             ratio=aspect_ratios[index]
#             R=generate_rect(r,c,scale,ratio)
#             RS = calculate_surrounding_rectangle(R)
#             if (check_that_rectangle_is_inside(RS,w,h)):
#                 hist_R=calculate_histogram_fast(labeled_map,inte_hist,R)
#                 hist_RS_temp=calculate_histogram_fast(labeled_map,inte_hist,RS)
#                 hist_RS=hist_RS_temp-hist_R
#                 distance=chi_square_distance(hist_R,hist_RS)
#                 if (distance > max_chi_square_distance):
#                     max_chi_square_distance = distance
#                     best_rect = R
#     if (final>=initial):
#         saliency_value=max_chi_square_distance
#         most_salient_rectangle=best_rect
#     else:
#         saliency_value=0
#         most_salient_rectangle=np.array([0,0,0,0])
#
#     return saliency_value,most_salient_rectangle

# def distance_map(img):
#     img=img.astype(float)
#     h,w,c=img.shape
#     map=np.zeros((h,w))
#     rectangle=np.zeros((h,w,4))
#     labeled_map=rgb_labeled_map(img)
#     inte_hist=integral_histogram(labeled_map)
#     for i in range(0,h,5):
#         for j in range(0,w,5):
#             distance,rect=saliency_at_position(labeled_map,inte_hist,i,j)
#             map[i,j]=distance
#             rectangle[i,j]=rect
#     map=(map-np.min(map))/(np.max(map)-np.min(map))
#     return map,rectangle

# def center_surround_histogram_map(distances, rectangles):
#     h,w=distances.shape
#     map=np.zeros((h,w))
#     for i in range(0,h,5):
#         for j in range(0,w,5):
#             most_distinctive_rectangle=rectangles[i,j]
#             r_left,r_top,r_width,r_height=encompass(most_distinctive_rectangle, [1,1,w,h])
#             chi_square_at_this_pixel = distances[i,j]
#             temp=r_height * r_width
#             if temp==0:
#                 temp=temp+float('inf')
#             inverse_sigma_squared = 3 / temp
#             for idy in range(int(r_top),int(r_top+r_height)):
#                 for idx in range(int(r_left),int(r_left+r_width)):
#                     deltay = idy - i
#                     deltax = idx - j
#                     euclidean_distance_squared = deltay * deltay + deltax * deltax
#                     weight = math.exp(-0.5 * euclidean_distance_squared * inverse_sigma_squared)
#                     map[idy, idx] =map[idy, idx]+ weight * chi_square_at_this_pixel
#     map=(map-np.min(map))/(np.max(map)-np.min(map))
#     return map

def EM_init_kmeans(Data, nbStates):
    nbVar, nbData = Data.shape
    estimator=KMeans(nbStates)
    estimator.fit(Data.T)
    Data_id=estimator.labels_
    Centers=estimator.cluster_centers_
    # inertia=estimator.inertia_
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
        pix=pix_tmp/(temper.T)
        E=sum(pix)
        for i in range(0,nbStates):
            priors[0,i]=E[i]/nbData
            mu[:,i]=np.dot(data,pix[:,i])/E[i]
            data_tmp1=data-(np.tile(mu[:,i],(nbData,1))).T
            sigma[:,:,i]=np.dot(np.tile(pix[:,i].T,(nbVar,1))*data_tmp1,(data_tmp1.T))/E[i]
            sigma[:,:,i]=sigma[:,:,i]+1e-5*np.diag(np.ones((nbVar,1))[0])
        for i in range(0,nbStates):
            pix[:,i]=gaussPDF(data,mu[:,i],sigma[:,:,i])
        F=np.dot(pix,priors.T)
        F[np.where(F<2.2251e-308)]=2.2251e-308
        loglik=np.mean(np.log(F))
        if np.fabs((loglik/loglik_old)-1)<loglike_threshold:
            break
        loglik_old=loglik
        nbStep=nbStep+1
    for i in range(0, nbStates):
        sigma[:, :, i] = sigma[:, :, i] + 1e-5 * np.eye((nbVar))
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

# def calculate_horizontal_variances(colormaps):
#     height, width, numberOfColors=colormaps.shape
#     colormap_sum=np.zeros((numberOfColors,1))
#     for i in range(0,numberOfColors):
#         colormap_sum[i]=sum(sum(colormaps[:,:,i]))
#     M_h=np.zeros((numberOfColors,1))
#     for color in range(0,numberOfColors):
#         for y in range(0,height):
#             for x in range(0,width):
#                 M_h[color]=M_h[color]+colormaps[y,x,color]*x
#     M_h=M_h/colormap_sum
#     horizontal_variances=np.zeros((numberOfColors,1))
#     for y in range(0,height):
#         for x in range(0,width):
#             for color in range(0,numberOfColors):
#                 temp=np.fabs(x-M_h[color])
#                 horizontal_variances[color]=horizontal_variances[color]+colormaps[y,x,color]*temp*temp
#     horizontal_variances=horizontal_variances/colormap_sum
#     variances=horizontal_variances
#     return variances

# def calculate_vertical_variances(colormaps):
#     height, width, numberOfColors = colormaps.shape
#     colormap_sum = np.zeros((numberOfColors, 1))
#     for i in range(0, numberOfColors):
#         colormap_sum[i] = sum(sum(colormaps[:, :, i]))
#     M_v = np.zeros((numberOfColors, 1))
#     for color in range(0, numberOfColors):
#         for y in range(0, height):
#             for x in range(0, width):
#                 M_v[color] = M_v[color] + colormaps[y, x, color] * y
#     M_v = M_v / colormap_sum
#     vertical_variances = np.zeros((numberOfColors, 1))
#     for y in range(0, height):
#         for x in range(0, width):
#             for color in range(0, numberOfColors):
#                 temp = np.fabs(y - M_v[color])
#                 vertical_variances[color] = vertical_variances[color] + colormaps[y, x, color] * temp * temp
#     vertical_variances = vertical_variances / colormap_sum
#     variances = vertical_variances
#     return variances

def calculate_spatial_variances(colormaps):
    horizontal_variances = EFM.calculate_horizontal_variances(colormaps)
    vertical_variances = EFM.calculate_vertical_variances(colormaps)
    variances = horizontal_variances + vertical_variances
    variances = (variances - np.min(variances[:]))/ (np.max(variances[:]) - np.min(variances[:]))
    return variances

# def cw_csd_map(colormaps):
#     height, width, numberOfColors=colormaps.shape
#     output_map=np.zeros((height,width))
#     variance=calculate_spatial_variances(colormaps)
#     center_distance_weights = np.zeros((numberOfColors, 1))
#     pixels_in_cluster = 0
#     epsilon = 0.00001
#     for color in range(0,numberOfColors):
#         for y in range(0,height):
#             for x in range(0,width):
#                 color_probability=colormaps[y,x,color]
#                 if (color_probability>(0+epsilon)):
#                     pixels_in_cluster+=1
#                 deltay=y-height/2
#                 deltax=x-width/2
#                 distance=np.sqrt(deltay*deltay+deltax*deltax)
#                 center_distance_weights[color]=center_distance_weights[color]+colormaps[y,x,color]*distance
#     center_distance_weights=(center_distance_weights-np.min(center_distance_weights))/(np.max(center_distance_weights)-np.min(center_distance_weights))
#     for color in range(0,numberOfColors):
#         output_map=output_map+(1-variance[color])*(1-center_distance_weights[color])*colormaps[:,:,color]
#     output_map=(output_map-np.min(output_map))/(np.max(output_map)-np.min(output_map))
#     return output_map
