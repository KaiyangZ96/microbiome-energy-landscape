# This is modified dijkstra for EL, not the original dijkstra algorithm
import numpy as np
import numpy.matlib
import pandas as pd
import scipy.cluster.hierarchy
import dendrogram2
import matplotlib.pyplot as plt
import os

#def Disconnectivity(E,LocalMinIndex,AdjacentList):
def Disconnectivity(E, LocalMinIndex, AdjacentList):
    Checkpoints = [str(i) for i in LocalMinIndex.ravel().tolist()] # transfer LocalMinIndex array to string list
    sy,sx = np.shape(AdjacentList)
    AdjacencyMatrix = np.zeros((sy,sy),dtype = bool)
    tmp1 = np.matlib.repmat(np.array(range(1,sy+1)), sx, 1) 
    tmp1.ravel('F')
    tmp2 = AdjacentList.T
    AdjacencyMatrix[tmp1.ravel('F')-1,tmp2.ravel('F')-1] = True # -1 for index in numpy 
    tmp3 = np.zeros((sy,sx))
    for i in range(1,sx):
        tmp3[:,i] = np.concatenate((E[AdjacentList[:,0]-1],E[AdjacentList[:,i]-1]),axis=1).max(1) # the cost Eaa' = max(Ea,Ea')
    N= np.empty((sy,sy),dtype = float)
    N.fill(np.inf)
    N[tmp1.ravel('C')-1,tmp2.ravel('C')-1] = tmp3.ravel('F') # fill distance matrix
    index = [str(i) for i in range(1,sy+1)]
    N_df = pd.DataFrame(N,index=index,columns=index)
    cost_matrix = np.zeros((len(Checkpoints),len(Checkpoints)))
    route_matrix = pd.DataFrame(cost_matrix)
    for ind_1,start in enumerate(Checkpoints):
        S = [start]
        U = dict(N_df[start])
        route = dict.fromkeys(U.keys(),list())
        while len(S) < len(N_df):
            S,U,route=update_dis(S,U,route,N_df)
        for ind_2,end in enumerate(Checkpoints):
            cost_matrix[ind_1,ind_2] = U[end]
            if start != end:
                route_matrix.iloc[ind_1,ind_2] = start+'_'+'_'.join(route[end])+'_'+end
            else:
                route_matrix.iloc[ind_1,ind_2] = start
    cost_matrix =  pd.DataFrame(cost_matrix)
    return cost_matrix,route_matrix # cost and route


def update_dis(S,U,route,dis_matrix): # key update part of modified dijkstra algorithm
    U_exclude_S= {key:value for key,value in U.items() if key not in S}
    new_S_node = min(U_exclude_S, key=U_exclude_S.get)
    dirct_new = dict(dis_matrix[new_S_node])
    S.append(new_S_node)
    connected_nodes = [node for node,dis in dirct_new.items() if dis != np.inf]
    check_update_nodes = [node for node in connected_nodes if node not in S]
    for node in check_update_nodes :
        if U[node]==np.inf or U[node]>= U[new_S_node]:
            U[node] = max(dirct_new[node],U[new_S_node])
            tmp_route = route[new_S_node].copy()
            tmp_route.append(new_S_node)
            route[node] = tmp_route
    return S,U,route


def drawDisconnectivityGraph(cost_matrix, localMinIndex, Energy, save_path):
    dis = np.tril(cost_matrix)[np.nonzero(np.tril(cost_matrix))]
    Z = scipy.cluster.hierarchy.linkage(dis)
    fig, ax = plt.subplots(figsize=(15,15))
    ax.set_ylabel('Energy')
    ax.set_xlabel('Local minimal parttern')
    labelList = list(localMinIndex.ravel())
    a = dendrogram2.dendrogram2(Z,data=Energy[[i-1 for i in labelList],:],labels=labelList)
    plt.title('Disconnectivity Graph', fontdict={'fontsize':20,'fontweight':'bold'}, loc='center', pad=20)
    plt.savefig(os.path.join(save_path,'DisconnectivityGraph.png'),format='png', bbox_inches='tight')
    plt.close()
    return