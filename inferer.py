
import numpy as np
def Inferrer_ML(binarizedData):
    [nodeNumber,dataLength] = np.shape(binarizedData)
    iterationMax = 5000000
    permissibleErr = 0.00000001
    dt = 0.2
    dataMean = np.mean(binarizedData,1)
    dataMean = dataMean.reshape(-1,1)
    dataCorrelation = binarizedData.dot(binarizedData.T)/dataLength
    h = np.zeros((nodeNumber,1))
    J = np.zeros((nodeNumber,nodeNumber))
    vectorlist = VectorList(nodeNumber)
    for i in range(iterationMax):
        [modelMean, modelCorrelation] = ModelMeanCorrelation(h,J,vectorlist);
        dh = dt*(dataMean-modelMean)
        dJ = dt*(dataCorrelation - modelCorrelation)
        dJ = dJ - np.diag(np.diag(dJ)) # make sure Jii=0
        h = h+dh
        J = J+dJ
        if np.sqrt(np.linalg.norm(dJ,'fro')**2 + np.linalg.norm(dh)**2)/nodeNumber/(nodeNumber+1) < permissibleErr:
            break
    return h,J

def ModelMeanCorrelation(h,J,vectorlist):
    nodeNumber = np.shape(h)[0]
    prob = StateProb(h,J,vectorlist)
    modelMean = (np.mean(vectorlist*np.ones((nodeNumber,1)).dot(prob.T),1)*(2**nodeNumber)).reshape(-1,1)
    modelCorrelation = (vectorlist*np.ones((nodeNumber,1)).dot(np.sqrt(prob.T))).dot((vectorlist*np.ones((nodeNumber,1)).dot(np.sqrt(prob.T))).T)
    return modelMean,modelCorrelation

def VectorList(nodeNum):
    node = 2**nodeNum
    array =  np.array(range(0,node))
    bin_lenth = len(format(node-1, 'b'))
    bin_array = list()
    for i in array:
        tmp = list(format(i,'0%sb'%bin_lenth))
        bin_array.append([{'0':-1,'1':1}[i] for i in tmp])
        
    return np.flipud(np.array(bin_array).T)

def StateProb(h,J,vectorList):
    nodeNumber = np.shape(h)[0]
    numVec = np.shape(vectorList)[1]
    Z = np.sum(np.exp(-(-np.diag(((0.5*J.dot(vectorList)).T).dot(vectorList)).reshape(-1,1)-np.sum((h.dot(np.ones((1,numVec)))*vectorList).T,1).reshape(-1,1))))
    probMEM = np.exp(-(-np.diag(((0.5*J.dot(vectorList)).T).dot(vectorList)).reshape(-1,1)-np.sum((h.dot(np.ones((1,numVec)))*vectorList).T,1).reshape(-1,1)))/Z
    return probMEM