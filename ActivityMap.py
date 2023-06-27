import pandas as pd
import inferer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors
import matplotlib.ticker
import os
def ActivityMap(nodeNum,localMinIndex,target_path,save_path):
    localMinIndex = list(localMinIndex.ravel())
    rownames = pd.read_csv(os.path.join(target_path,'rownames.txt'),header= None)
    rownames = list(rownames[0])
    vectorList = inferer.VectorList(nodeNum)
    localMinNum = len(localMinIndex)
    X = np.zeros((nodeNum,localMinNum))
    for i in range(localMinNum):
        X[:,i] = vectorList[:,localMinIndex[i]-1]
    row_number = nodeNum
    col_number = localMinNum
    fig, ax = plt.subplots()
    cmap = matplotlib.colors.ListedColormap(['grey','white'])
    bounds = [-1,0,1]
    norm = matplotlib.colors.BoundaryNorm(bounds,cmap.N)
    ax.imshow(X,cmap=cmap,norm=norm,extent=[0,col_number*2,row_number,0])
    ax.set_xticks([i+1 for i in range(0,col_number*2,2)]) # xtick location
    ax.set_xticklabels(localMinIndex)
    ax.set_yticks([i+.5 for i in range(0,row_number)])
    ax.set_yticklabels(rownames)
    # Minor ticks
    ax.set_yticks(np.arange(0, row_number, 1), minor=True)
    ax.set_xticks(np.arange(0, col_number*2, 2), minor=True)
    ax.grid(c='black',which='minor')

    ax.set_xlabel('local minimal pattern')
    ax.set_ylabel('rowname')
    plt.savefig(os.path.join(save_path, 'ActivityMap.png'),format = 'png')