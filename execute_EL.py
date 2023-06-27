# Estimate the parameters h and J in the pairwise MaxEnt model
import inferer
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Argument for execute_EL')
parser.add_argument('--target_path', '-t', help='target path includes binarized input file', default='./')
parser.add_argument('--binarized_file', '-b', help='binarized file name', required=True)
parser.add_argument('--save_path', '-s', help='output files\' save path', default='./')
args = parser.parse_args()
target_path = args.target_path
binarized_file = args.binarized_file
save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data and infer the parameters of the model
binarizedData = np.loadtxt(os.path.join(target_path,binarized_file),delimiter=',')
h,J = inferer.Inferrer_ML(binarizedData)
import pandas as pd
h_df  = pd.DataFrame(h)
J_df = pd.DataFrame(J)
h_df .to_csv(os.path.join(save_path, 'h.csv'), index=None, header=None)
J_df.to_csv(os.path.join(save_path, 'J.csv'), index=None, header=None)

# Calculate the energy of each pattern
import energy
E = energy.energy(h,J)
E_df = pd.DataFrame(E)
E_df.to_csv(os.path.join(save_path, 'E.csv'), index=None, header=None)
nodeNumber = np.shape(h)[0]


# Save the intermediate variables
import pickle 
with open(os.path.join(save_path,'tmp_obj.txt'), 'wb') as f:
    pickle.dump((h, J, E), f)


# Calculate local minima
import LocalMin
[LocalMinIndex,BasinGraph,AdjacentList] = LocalMin.LocalMin(nodeNumber, E)
LocalMinIndex_df = pd.DataFrame(LocalMinIndex)
BasinGraph_df = pd.DataFrame(BasinGraph)
AdjacentList_df = pd.DataFrame(AdjacentList)
LocalMinIndex_df.to_csv(os.path.join(save_path,'LocalMinIndex.csv'), index=None, header=None)
BasinGraph_df.to_csv(os.path.join(save_path,'BasinGraph.csv'), index=None, header=None)
AdjacentList_df.to_csv(os.path.join(save_path,'AdjacentList.csv'), index=None, header=None)

# Calculate disconnectivity by modified dijkstra method
import dijkstra
[Cost, Path] = dijkstra.Disconnectivity(E, LocalMinIndex, AdjacentList)
Cost.to_csv(os.path.join(save_path,'Cost.csv'), index=None, header=None)
Path.to_csv(os.path.join(save_path,'Path.csv'), index=None, header=None)
Z = dijkstra.drawDisconnectivityGraph(cost_matrix = Cost, localMinIndex=LocalMinIndex, Energy=E, save_path=save_path)

# Plot the activity map
import ActivityMap
ActivityMap.ActivityMap(nodeNumber, LocalMinIndex, target_path=target_path, save_path=save_path)

# Plot the local minima and basin (energy landscape plot)
import updateBasinList
EL_network_obj = updateBasinList.EL_network(LocalMinIndex, BasinGraph,E)
EL_network_obj.get_energyGroup()
EL_network_obj.draw_basinList(save_path=save_path)
EL_network_obj.draw_basinPlot_2D(save_path=save_path)
EL_network_obj.draw_basinPlot_3D(save_path=save_path)


