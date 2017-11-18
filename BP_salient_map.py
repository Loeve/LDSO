import numpy as np

def edgeNum(i,j,starV,starE):
     for k in range(starV[i],starV[i+1]):
        if j==starE[k]:
            return k
        return 0

def absoluteDif(n1,n2):
    if n1>n2:
        return n1-n2
    else:
        return n2-n1

def BP_infer(nodePot,edgePot,infEng,starV,starE):
    nNodes,nStates=nodePot.shape
    nEdges=edgePot.shape[2]
    # nodePot=np.reshape(nodePot,(nNodes*nStates,1))
    # edgePot=np.reshape(edgePot,(nStates*nStates*nEdges,1))
    # maximize=infEng.maximize
    maxIter=infEng.maxIter
    # maxIter=10
    # optTol=infEng.tol
    for i in range(0,nNodes+1):
        starV[i]=starV[i]-1
    for i in range(0,nEdges):
        starE[i]=starE[i]-1
    # # nodeBel=np.array((nNodes*nStates,1))
    # # msgs=np.array((nEdges*nStates,1))
    # # edgeBel=np.array((nStates*nStates*nEdges,1))
    # prod_of_msgs=np.zeros((nNodes,nStates))
    # old_bel=np.zeros((nNodes,nStates))
    # new_bel=np.zeros((nNodes,nStates))
    # old_msg=np.zeros((nEdges,nStates))
    # new_msg=np.zeros((nEdges,nStates))
    # newm=np.zeros((nStates,1))
    # pot_ij=np.zeros((nStates*nStates,1))
    # temp=np.zeros((nStates))
    # # prod_of_msgs=nodePot
    # # old_bel=nodePot
    # for i in range(0,nNodes):
    #     for j in range(0,nStates):
    #         prod_of_msgs[i,j]=nodePot[i+nNodes*j]
    #         old_bel[i,j]=nodePot[i+nNodes*j]
    # for i in range(0,nEdges):
    #     for j in range(0,nStates):
    #         old_msg[i,j]=1.0/nStates
    iter=0
    converged=0

    while(converged==0)and (iter<maxIter):
        for i in range(0,nNodes):
            nbrsInd=starV[i]
            nbrs=starE[nbrsInd:]
            nNbrs=starV[i+1]-starV[i]

            for j in range(0,nNbrs):
                # if (i<nbrs[j]):
                #     for ii in range(0,nStates):
                #         for jj in range(0,nStates):
                #             pot_ij[ii+nStates*jj]=edgePot[ii+nStates*(jj+nStates*(nbrsInd+j))]
                #             # pot_ij[ii,jj]=edgePot[ii,jj,nbrsInd+j]
                # else:
                #     for ii in range(0,nStates):
                #         for jj in range(0,nStates):
                #             pot_ij[jj+nStates*ii]=edgePot[ii+nStates*(jj+nStates*(nbrsInd+j))]
                #             # pot_ij[jj,ii]=edgePot[ii,jj,nbrsInd+j]
                # for ii in range(0,nStates):
                #     temp[ii]=nodePot[i+ii*nNodes]
                #     # temp[ii] = nodePot[i,ii]
                for k in range(0,nNbrs):
                    if(k==j):
                        continue
                    e=edgeNum(nbrs[k],i,starV,starE)
                    # for ii in range(0,nStates):
                    #     temp[ii]=temp[ii]*old_msg[e,ii]
                # if(maximize):
                #     print("BP_General_C only supports sum-product")
                # else:
                #     for ii in range(0,nStates):
                #         newm[ii]=0
                #         for iii in range(0,nStates):
                #             newm[ii]+=pot_ij[iii+nStates*ii]*temp[iii]
                #             # newm[ii] += pot_ij[iii,ii] * temp[iii]
                #     summ=0
                #     for ii in range(0,nStates):
                #         summ+=newm[ii]
                #     for ii in range(0,nStates):
                #         new_msg[nbrsInd+j,ii]=newm[ii]/summ

        # for i in range(0,nNodes):
        #     nbrsInd=starV[i]
        #     nbrs = starE[nbrsInd:]
        #     nNbrs = starV[i + 1] - starV[i]
        #     for j in range(0,nStates):
        #         prod_of_msgs[i,j]=nodePot[i+j*nNodes]
        #         # prod_of_msgs[i, j] = nodePot[i,j]
        #     for j in range(0,nNbrs):
        #         e=edgeNum(nbrs[j],i,starV,starE)
        #         for k in range(0,nStates):
        #             prod_of_msgs[i,k]=prod_of_msgs[i,k]*new_msg[e,k]
        #     summ=0
        #     for j in range(0,nStates):
        #         summ+=prod_of_msgs[i,j]
        #     for j in range(0,nStates):
        #         new_bel[i,j]=prod_of_msgs[i,j]/summ

        # converged=1
        # for i in range(0,nNodes):       #可以优化为最小值和阈值比较
        #     for j in range(0,nStates):
        #         if(absoluteDif(new_bel[i,j],old_bel[i,j])>optTol):
        #             converged=0
        iter+=1
        # for i in range(0,nNodes):
        #     for j in range(0,nStates):
        #         old_bel[i,j]=new_bel[i,j]
        # for i in range(0,nEdges):
        #     for j in range(0,nStates):
        #         old_msg[i,j]=new_msg[i,j]

    # if(converged!=1):
    #     print(["BP: Stopped after maxIter = %d iterations" % iter])
    nodeBel = np.zeros((nNodes * nStates, 1))
    # for i in range(0,nNodes):
    #     for j in range(0,nStates):
    #         nodeBel[i+nNodes*j]=new_bel[i,j]


    return nodeBel