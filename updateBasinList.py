import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import os

class EL_network():
    """ class for generation of 2D and 3D energy landscape plot as well as definition of basin groups"""

    def __init__(self,LocalMinIndex,BasinGraph,E):
        self.LocalMinIndex = list(LocalMinIndex.ravel())
        self.E = E
        G = nx.DiGraph()
        G.add_nodes_from(BasinGraph[:,0])
        G.add_edges_from(BasinGraph)
        self.G = G

    def get_energyGroup(self):
        """ get weakly connected component of node""" 
        core_node_dict = dict()
        for cc in nx.weakly_connected_components(self.G):
            core_node = set(self.LocalMinIndex).intersection(set(cc))
        
            if len(core_node) > 1:
                raise Exception("Check the localMin!", core_node)
            elif not core_node:
                next()
            else:
                core_node = list(core_node)[0]
                core_node_dict[core_node] = cc
        self.BasinGroup = core_node_dict
        return

    def draw_basinList(self,save_path):
        with open(os.path.join(save_path,'BasinList.txt'),'w') as f:
            for i in self.BasinGroup.items():
                f.write(str(i[0])+':'+str(i[1])+'\n')
        return

    def draw_basinPlot_2D(self,save_path):
        plt.figure(dpi=100,figsize=(10,10))
        #pos = nx.spring_layout(self.G,dim=2,seed = 3008)
        pos = nx.nx_agraph.graphviz_layout(self.G, prog="fdp") # need  graphviz and pygraphviz
        cmap = plt.cm.viridis
        nx.draw(self.G,pos = pos, with_labels=False,cmap=cmap,node_color = self.E,node_size=30,font_size=10,edge_color='skyblue')
        # draw label
        pos_shift = {} 
        x_off = .05  # offset on the x axis
        for k, v in pos.items():
            pos_shift [k] = (v[0]+x_off, v[1])
        nx.draw_networkx_labels(self.G, pos_shift )
        # plot color bar
        vmin = min(self.E)
        vmax = max(self.E)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
        plt.colorbar(sm)
        plt.axis('on')
        plt.title('Local Minima and Basin(2D)', fontdict={'fontsize':20,'fontweight':'bold'}, loc='center')
        plt.savefig(os.path.join(save_path,'LocalMinimaBasin2D.png'), format='png', bbox_inches='tight') # bbox_inches='tight' avoids cutting the title
        #plt.show()
        plt.close()
        return

    def draw_basinPlot_3D(self,save_path):

        def front_rear_edge(edge_xyz):
            tmp_start = edge_xyz[:,0,:] 
            tmp_end = edge_xyz[:,1,:] 
            tmp_middle = (edge_xyz[:,1,:]+edge_xyz[:,0,:])/2
            front_edge_xyz= np.array([(tmp_start[i],tmp_middle[i]) for i in range(len(tmp_start))])
            rear_edge_xyz= np.array([(tmp_middle[i],tmp_end[i]) for i in range(len(tmp_start))])
            return front_edge_xyz,rear_edge_xyz


        class Arrow3D(FancyArrowPatch):

            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
                self._verts3d = xs, ys, zs


            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
                FancyArrowPatch.draw(self, renderer)

            do_3d_projection = draw

        # Set up 3D node and edge layout
        pos = nx.nx_agraph.graphviz_layout(self.G, prog="fdp") # need  graphviz and pygraphviz
        node_xyz = np.array([np.append(pos[v],self.E[v-1]) for v in sorted(self.G)])
        edge_xyz = np.array([(np.append(pos[u],self.E[u-1]), np.append(pos[v],self.E[v-1])) for u, v in self.G.edges()])
        # Create the 3D figure
        fig = plt.figure(dpi=100,figsize=(10,10))
        from mpl_toolkits.mplot3d import Axes3D
        ax = Axes3D(fig)
        ax.view_init(elev=20., azim=45) # set the angle
        # Plot the nodes
        cmap = plt.cm.viridis
        ax.scatter(*node_xyz.T, s=20, ec=None,alpha =1,cmap = cmap,c = self.E)
        for i in range(len(node_xyz)):
            ax.text(node_xyz[i,0],node_xyz[i,1]+0.03,node_xyz[i,2],'%s' % (str(i+1)), size=10, zorder=1,  color='k') # shift the label location
        # Plot the edges
        front_edge_xyz,rear_edge_xyz = front_rear_edge(edge_xyz)
        for vizedge in front_edge_xyz:
            arrowstyle = mpl.patches.ArrowStyle("->, head_length=0.1, head_width=0.1")
            a = Arrow3D(*vizedge.T, mutation_scale=20, 
                        lw=0.5, arrowstyle=arrowstyle, color="skyblue",alpha=0.7)
            ax.add_artist(a)
        for vizedge in rear_edge_xyz:
            ax.plot([vizedge[0][0], vizedge[1][0]], [vizedge[0][1], vizedge[1][1]], [vizedge[0][2], vizedge[1][2]], color="skyblue", alpha=0.7, lw=0.5)

        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines off
            #ax.grid(False)
            # Suppress tick labels
            for dim in (ax.xaxis, ax.yaxis):
                dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("Energy")

        # Add color bar
        vmin = min(self.E)
        vmax = max(self.E)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
        plt.colorbar(sm,fraction=0.046, pad=0.04)
        plt.title('Local Minima and Basin(3D)', fontdict={'fontsize':20,'fontweight':'bold'}, loc='center')
        _format_axes(ax)
        #fig.tight_layout()
        plt.savefig(os.path.join(save_path,'LocalMinimaBasin3D.png'), format='png', bbox_inches = "tight")
        #plt.show()
        plt.close()
        return


