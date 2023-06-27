import numpy as np
def energy(h,J):
    nodeNumber = np.shape(h)[0]
    vectorlist = VectorList(nodeNumber)
    numVec = np.shape(vectorlist)[1]
    E = -np.diag(((0.5*J).dot(vectorlist).T).dot(vectorlist))-np.sum(h.dot(np.ones((1,numVec)))*vectorlist,0)
    E = E.reshape(-1,1)
    return E
def VectorList(nodeNum):
    node = 2**nodeNum
    array =  np.array(range(0,node))
    bin_lenth = len(format(node-1, 'b'))
    bin_array = list()
    for i in array:
        tmp = list(format(i,'0%sb'%bin_lenth))
        bin_array.append([{'0':-1,'1':1}[i] for i in tmp]) 
    return np.flipud(np.array(bin_array).T)