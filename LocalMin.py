import numpy as np
def VectorList(nodeNum):
    node = 2**nodeNum
    array =  np.array(range(0,node))
    bin_lenth = len(format(node-1, 'b'))
    bin_array = list()
    for i in array:
        tmp = list(format(i,'0%sb'%bin_lenth))
        bin_array.append([{'0':-1,'1':1}[i] for i in tmp])
        
    return np.flipud(np.array(bin_array).T)
def VectorIndex(vectorList):
    index_list = list()
    for i in vectorList.T:
        tmp = i
        tmp = [{-1:'0',1:'1'}[i] for i in tmp]
        tmp = ''.join(tmp)
        index_list.append(int(tmp, 2)+1)
    EnergyIndex = np.array(index_list).reshape(-1,1)
    vectorIndex = EnergyIndex
    return vectorIndex,EnergyIndex

def LocalMin(nodeNumber,E):
    vectorList = VectorList(nodeNumber)
    [vectorIndex,EnergyIndex] = VectorIndex(vectorList)
    NeighborMatrix = vectorIndex
    #Calculate local minimum points

    #Make nearest neighbor index matrix
    #Each data is one index different from original position
    Loc = 1
    for i in range(1,nodeNumber+1):
        tmp = np.bitwise_xor(vectorIndex-1,Loc)+1 #find neighbour index
        NeighborMatrix = np.c_[NeighborMatrix,tmp]; # add column
        Loc = Loc*2
    #Calculate adjacent list for adjacency matrix
    AdjacentList = EnergyIndex[NeighborMatrix-1,0] 
    #Make energy list at each position including neighbors
    NeighborEnergy = E[AdjacentList-1,0]
    #Calculate minimum energy and its index
    EnergyMin = np.min(NeighborEnergy,1)
    EnergyMin_idx = list() # from zero
    for i in range(len(EnergyMin)):
        EnergyMin_idx.append(np.where(NeighborEnergy[i] == EnergyMin[i])[0]) # find minimun energy in each nighbour patterns(row)
    LocalMinIndex = AdjacentList[np.where(np.array(EnergyMin_idx)==0)].reshape(-1,1)
    idx = AdjacentList[:,0]
    BasinGraph = AdjacentList[np.array(range(0,2**nodeNumber)),np.array(EnergyMin_idx).reshape(1,-1)].reshape(-1,1)
    BasinGraph = np.c_[idx,BasinGraph]
    return LocalMinIndex,BasinGraph,AdjacentList
