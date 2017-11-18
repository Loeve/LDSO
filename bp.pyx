import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def BP_infer(np.ndarray[np.float64_t, ndim=2] nodePot0,np.ndarray[np.float64_t, ndim=3] edgePot0,np.ndarray[int, ndim=1] starV,np.ndarray[int, ndim=1] starE,int maximize,int maxIter,float optTol):
    cdef int i,j,k,ii,jj,iii,nbrsInd,nbrsInd2,e,iter,n,k_i,k_j
    cdef int nNodes,nEdges,nStates,converged,nNbrs
    cdef float summ
    nNodes=nodePot0.shape[0]
    nStates=nodePot0.shape[1]
    nEdges=edgePot0.shape[2]
    cdef np.ndarray[int,ndim=1] nbrs
    cdef np.ndarray[np.float64_t,ndim=1] nodePot=np.reshape(nodePot0,(nNodes*nStates))
    cdef np.ndarray[np.float64_t,ndim=1] edgePot=np.reshape(edgePot0,(nEdges*nStates*nStates))
    cdef np.ndarray[np.float64_t,ndim=2] prod_of_msgs=np.zeros((nNodes,nStates),dtype=np.float)
    cdef np.ndarray[np.float64_t,ndim=2] old_bel=np.zeros((nNodes,nStates),dtype=np.float)
    cdef np.ndarray[np.float64_t,ndim=2] new_bel=np.zeros((nNodes,nStates),dtype=np.float)
    cdef np.ndarray[np.float64_t,ndim=2] old_msg=np.zeros((nEdges,nStates),dtype=np.float)
    cdef np.ndarray[np.float64_t,ndim=2] new_msg=np.zeros((nEdges,nStates),dtype=np.float)
    cdef np.ndarray[np.float64_t,ndim=1] newm=np.zeros((nStates),dtype=np.float)
    cdef np.ndarray[np.float64_t,ndim=1] pot_ij=np.zeros((nStates*nStates),dtype=np.float)
    cdef np.ndarray[np.float64_t,ndim=1] temp=np.zeros((nStates),dtype=np.float)
    cdef np.ndarray[np.float64_t,ndim=2] nodeBel=np.zeros((nNodes,nStates),dtype=np.float)
#    cdef np.ndarray[np.float64_t,ndim=1] msgs=np.zeros((nEdges*nStates),dtype=np.float)
#    cdef np.ndarray[np.float64_t,ndim=3] edgeBel=np.zeros((nStates,nStates,nEdges),dtype=np.float)

    for i in range(0,nNodes+1):
        starV[i]=starV[i]-1
    for i in range(0,nEdges):
        starE[i]=starE[i]-1

    for i in range(0,nNodes):
        for j in range(0,nStates):
            prod_of_msgs[i,j]=nodePot0[i,j]
            old_bel[i,j]=nodePot0[i,j]

    for i in range(0,nEdges):
        for j in range(0,nStates):
            old_msg[i,j]=1.0/nStates
    iter=0
    converged=0
  #  print(edgePot0[:,:,:10])
  #  print(nodePot0[:20,:])
    while(converged==0)and (iter<maxIter):
        for i in range(0,nNodes):
            nbrsInd=starV[i]
            nbrs=starE[nbrsInd:]
            nNbrs=starV[i+1]-starV[i]
            for j in range(0,nNbrs):
                if (i<nbrs[j]):
                    for ii in range(0,nStates):
                        for jj in range(0,nStates):
                            pot_ij[ii+nStates*jj]=edgePot0[ii,jj,(nbrsInd+j)]
                            #pot_ij[ii+nStates*jj]=edgePot[ii+nStates*(jj+nStates*(nbrsInd+j))]

                else:
                    for ii in range(0,nStates):
                        for jj in range(0,nStates):
                            pot_ij[jj+nStates*ii]=edgePot0[ii,jj,(nbrsInd+j)]
                            #pot_ij[jj+nStates*ii]=edgePot[ii+nStates*(jj+nStates*(nbrsInd+j))]

                for ii in range(0,nStates):
                    temp[ii]=nodePot0[i,ii]
                #if(i<4):
                #        print(i,j)
                #        print(newm)
                for k in range(0,nNbrs):
                    if(k==j):
                        continue
                    e=edgeNum(nbrs[k],i,starV,starE)
                    for ii in range(0,nStates):
                        temp[ii]=temp[ii]*old_msg[e,ii]
                if(maximize==0):
                    for ii in range(0,nStates):
                        newm[ii]=0
                        for iii in range(0,nStates):
                            newm[ii]+=pot_ij[iii+nStates*ii]*temp[iii]
                   # if(i<4):
                   #     print(i,j)
                   #     print(newm)

                    summ=0
                    for ii in range(0,nStates):
                        summ+=newm[ii]
                    for ii in range(0,nStates):
                        new_msg[nbrsInd+j,ii]=newm[ii]/summ
        #print(new_msg[:20,:])

        for i in range(0,nNodes):
            nbrsInd=starV[i]
            nbrs = starE[nbrsInd:]
            nNbrs = starV[i + 1] - starV[i]
            for j in range(0,nStates):
                prod_of_msgs[i,j]=nodePot0[i,j]
            #if(i<4):
            #    print(i,j)
            #    print(prod_of_msgs[i,:])

            for j in range(0,nNbrs):
                e=edgeNum(nbrs[j],i,starV,starE)
                for k in range(0,nStates):
                    prod_of_msgs[i,k]=prod_of_msgs[i,k]*new_msg[e,k]
            #if(i<4):
            #    print(i,j)
            #    print(prod_of_msgs[i,:])

            summ=0
            for j in range(0,nStates):
                summ+=prod_of_msgs[i,j]
            for j in range(0,nStates):
                new_bel[i,j]=prod_of_msgs[i,j]/summ
        #print(new_bel[:20,:])

        converged=1
        for i in range(0,nNodes):
            for j in range(0,nStates):
                if(absoluteDif(new_bel[i,j],old_bel[i,j])>optTol):
                    converged=0
        iter+=1
        for i in range(0,nNodes):
            for j in range(0,nStates):
                old_bel[i,j]=new_bel[i,j]
        for i in range(0,nEdges):
            for j in range(0,nStates):
                old_msg[i,j]=new_msg[i,j]

    for i in range(0,nNodes):
        for j in range(0,nStates):
            nodeBel[i ,j]=new_bel[i,j]
    #print(new_bel[:100,:])
    #print(nodeBel[29900:])

    print("iter,converged",iter,converged)

#    return starV,starE
    return nodeBel


def absoluteDif(float n1,float n2):
    if n1>n2:
        return n1-n2
    else:
        return n2-n1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int edgeNum(int i,int j,np.ndarray[int, ndim=1] starV,np.ndarray[int, ndim=1] starE):
    cdef int k
    for k in range(starV[i],starV[i+1]):
        if j==starE[k]:
            return k
        return 0