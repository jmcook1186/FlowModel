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
        satT = np.zeros(shape=(shape_x,shape_y))

        qE = np.zeros(shape=(shape_x,shape_y))
        qW = np.zeros(shape=(shape_x,shape_y))
        qN = np.zeros(shape=(shape_x,shape_y))
        qS = np.zeros(shape=(shape_x,shape_y))

        Flux = np.zeros(shape=(shape_x,shape_y))

        return ztop, zbot, h, K, qE, qW, qS, qN, satT, Flux


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


    def run_model(t0,t_lim,t_step,len_x,len_y, h, cell_spacing, K, qE, qW, qS, qN, melt, ztop, zbot, h_boundary, del_z, satT):
        

        for i in np.arange(1,len_x-1,cell_spacing):
            for j in np.arange(1,len_y-1,cell_spacing):
                
                # calculate saturated thickness. if head is below bottom surface, satT = 0. 
                # If head is above upper surface, satT is equal to WC thickness
                # If head is between the two, it is the vertical distance from bottom surface to head.

                if h[i,j] > ztop[i,j]:

                    satT[i,j] = del_z[i,j]

                elif h[i,j] <= zbot[i,j]:

                    satT[i,j] = 0
                
                else:
                    satT[i,j] = h[i,j] - zbot[i,j]


                # calculate internode hydraulic conductivity as arithmetic mean of cell and adjacent cell
                KE = (K[i,j] + K[i-1,j]) / 2
                KW = (K[i,j] + K[i+1,j]) / 2
                KN = (K[i,j] + K[i,j-1]) / 2
                KS = (K[i,j] + K[i,j+1]) / 2

                # calculate internode saturated thickness as arithmetic mean of cell and adjacent cell
                satTE = (satT[i,j] + satT[i-1,j] ) / 2
                satTW = (satT[i,j] + satT[i+1,j] ) / 2
                satTN = (satT[i,j] + satT[i,j-1] ) / 2
                satTS = (satT[i,j] + satT[i,j+1] ) / 2

                # calculate internode head as arithmetic mean of cell and adjacent cell
                hE = h[i-1,j] - h[i,j]
                hW = h[i+1,j] - h[i,j] 
                hN = h[i,j-1] - h[i,j] 
                hS = h[i,j+1] - h[i,j] 

                # calculate flux in each cartesian direction
                tempE = KE * (hE / cell_spacing) * cell_spacing * satTE
                tempW = KW * (hW / cell_spacing) * cell_spacing * satTW
                tempN = KN * (hN / cell_spacing) * cell_spacing * satTN
                tempS = KS * (hS / cell_spacing) * cell_spacing * satTS

                FluxTot = tempE + tempW + tempS + tempN
                
                h[i,j] = h[i,j] + FluxTot

                # h[i-1,j] = h[i-1,j] - tempE 
                # h[i+1,j] = h[i+1,j] - tempW
                # h[i,j-1] = h[i,j-1] - tempN
                # h[i,j+1] = h[i,j+1] - tempS

        h = h + melt

        print(FluxTot)              

        return h, satT



    def plot_grid(grid, cbar_max, cbar_min, savepath, zbot, ztop):
    
        X,Y = np.meshgrid(np.arange(0,grid.shape[0]-2,1),np.arange(0,grid.shape[1]-2,1))
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X,Y,grid[1:-1,1:-1])
        surf2 = ax.plot_surface(X,Y,ztop[1:-1,1:-1], color = 'w', alpha=0.99)
        ax.set_zlim(50, 110)
        plt.savefig(savepath)
        plt.close()

        return