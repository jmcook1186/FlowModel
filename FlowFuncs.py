import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FlowFuncs:

    def __init__():

        return

    def make_empty_grids(len_x, len_y, cell_spacing):

        shape_x = int(len_x/cell_spacing)
        shape_y = int(len_y/cell_spacing)

        ztop = np.zeros(shape=(shape_x,shape_y))
        zbot = np.zeros(shape=(shape_x,shape_y))
        h = np.zeros(shape=(shape_x,shape_y))
        K = np.zeros(shape=(shape_x,shape_y))

        qE = np.zeros(shape=(shape_x,shape_y))
        qW = np.zeros(shape=(shape_x,shape_y))
        qN = np.zeros(shape=(shape_x,shape_y))
        qS = np.zeros(shape=(shape_x,shape_y))

        return ztop, zbot, h, K, qE, qW, qS, qN


    def fill_param_grids(ztop, zbot, h, K, slope, WCthickness, initial_WT, h_boundary, melt_rate, K_magnitude):

        # UPPER SURFACE
        ztop = (np.random.rand(ztop.shape[0],ztop.shape[1])*2)+100 #initial elevation

        for i in range(ztop.shape[0]):
            ztop[:,i] = ztop[:,i] - i*slope

        # WC LOWER SURFACE (boundary with impermeable ice)
        zbot = ztop - WCthickness #(np.random.rand(ztop.shape[0],ztop.shape[1])*10)/2
        del_z = ztop - zbot

        # WATER TABLE HEIGHT
        h = zbot+(del_z * initial_WT) #  fill the aquifer
        h[[0,-1],:] = h_boundary
        h[:,[0,-1]] = h_boundary


        K = np.random.rand(K.shape[0],K.shape[1])*K_magnitude
        melt = np.zeros(shape=(ztop.shape[0],ztop.shape[1])) + melt_rate #(vol/time)

        return ztop, zbot, del_z, h, K, melt


    def run_model(t0,t_lim,t_step,len_x,len_y, h, cell_spacing, K, qE, qW, qS, qN, melt, ztop, zbot, h_boundary):

        for i in np.arange(1,len_x-1,cell_spacing):
            for j in np.arange(1,len_y-1,cell_spacing):

                # calculate rate of flow (flux) in each cartesion direction
                qE[i,j] = ((K[i-1,j] + K[i,j])/2) * cell_spacing**2 * ((h[i-1,j] - h[i,j]) / cell_spacing)
                qW[i,j] = ((K[i+1,j] + K[i,j])/2) * cell_spacing**2 * ((h[i+1,j] - h[i,j]) / cell_spacing)
                qN[i,j] = ((K[i,j-1] + K[i,j])/2) * cell_spacing**2 * ((h[i,j-1] - h[i,j]) / cell_spacing)
                qS[i,j] = ((K[i,j+1] + K[i,j])/2) * cell_spacing**2 * ((h[i,j+1] - h[i,j]) / cell_spacing)

                # total rate of flux from cell
                FluxTot = qE[i,j] + qW[i,j] + qN[i,j] + qS[i,j]
                
                h[i,j] = h[i,j] + ((FluxTot + melt[i,j]))

        return h



    def plot_grid(grid, cbar_max, cbar_min, savepath):
    
        X,Y = np.meshgrid(np.arange(0,grid.shape[0]-2,1),np.arange(0,grid.shape[1]-2,1))
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X,Y,grid[1:-1,1:-1])

        ax.set_zlim(50, 110)
        fig.colorbar(surf)
        plt.savefig(savepath)
        plt.close()

        return