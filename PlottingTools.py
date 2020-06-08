import matplotlib.pyplot as plt
import numpy as np
from ParticleModelFuncs import vector_arrows

def plotFigures(x, y, z, plot_types, Out, t, plot_layer, Cells, figsize, savepath):

    if plot_types != None:

        for plot_type in plot_types:
            
            if plot_type == 'Q':

                for i in range(len(t)-1):

                    plt.figure(figsize=figsize)
                    plt.title('Net flow into cells in layer {}'.format(plot_layer))
                    plt.imshow(Out.Q[i,plot_layer, 2:-2, 2:-2],vmin=-0.5,vmax=0.5)
                    plt.colorbar()
                    plt.savefig(str(savepath+'Net_inflow_at_t{}.png'.format(i)))
                    plt.close()

            if plot_type == 'Qs':
                
                for i in range(len(t)-1):
                    plt.figure(figsize=figsize)
                    plt.title('Net flow out of cells in layer {}'.format(plot_layer))
                    plt.imshow(Out.Qs[i,plot_layer, 2:-2, 2:-2])
                    plt.colorbar()
                    plt.savefig(str(savepath+'Net_outflow_at_t{}.png'.format(i)))
                    plt.close()

            if plot_type == 'Qx':
                
                for i in range(len(t)-1):
                    plt.figure(figsize=figsize)
                    plt.title('Lateral flow into cells in layer {}'.format(plot_layer))
                    plt.imshow(Out.Qx[i,plot_layer, 2:-2, 2:-2])
                    plt.colorbar()
                    plt.savefig(str(savepath+'Net_lateral_flow_at_t{}.png'.format(i)))
                    plt.close()

            if plot_type == 'Qy':
                
                for i in range(len(t)-1):
                    plt.figure(figsize=figsize)
                    plt.title('Net longitudinal flow into cells in layer {}'.format(plot_layer))
                    plt.imshow(Out.Qy[i,plot_layer, 2:-2, 2:-2],vmin=-1,vmax=1)
                    plt.colorbar()
                    plt.savefig(str(savepath+'Net_longitudinal_flow_at_t{}.png'.format(i)))
                    plt.close()

            if plot_type =='Phi':

                for i in range(len(t)-1):
                    plt.figure(figsize=figsize)
                    plt.title('Hydraulic head in cell centres in layer {}'.format(plot_layer))
                    plt.imshow(Out.Phi[i,plot_layer, 2:-2, 2:-2],vmin=-1,vmax=1)
                    plt.colorbar()
                    plt.savefig(str(savepath+'Hydraulic_Head_at_t{}.png'.format(i)))
                    plt.close()

            if plot_type == 'Phi3D':

                for i in range(len(t)-1):
                    X,Y = np.meshgrid(x[3:-2],y[3:-2])
                    Z = Out.Phi[i,plot_layer,2:-2,2:-2]
                    ZZ = glacier.upper_surface[2:-2,2:-2]
                    plt.figure(figsize=figsize)
                    ax = plt.axes(projection='3d')
                    ax.plot_surface(Y, X, Z, cmap='winter', edgecolor='none')
                    ax.plot_wireframe(Y,X,ZZ,color='k',alpha=0.2)
                    ax.set_title('Hydraulic Head at t{}'.format(i))
                    ax.set_zlim(90,130)
                    plt.savefig(str(savepath+'Hydraulic_Head_at_t{}.png'.format(i)))
                    plt.close()

            if plot_type == 'VectorArrows':

                # calculate flow vectors between cells
                X,Y,Z,U,V,W = vector_arrows(Out, x, y, z, plot_layer)
                color_array = np.arange(0,50,1)


                for t in np.arange(0,len(U[:,0,0]),1):
                    
                    Ut = U[t,1,:,:]
                    Vt = V[t,1,:,:]
                    plt.figure(figsize=figsize)
                    plt.quiver(X,Y,Ut,Vt,color_array, cmap='autumn',scale=10000)
                    plt.savefig(str(savepath + 'VectorFig{}.png'.format(t)))
            
            if plot_type == "Cells":
                
                for t in np.arange(0,len(Cells[:,0,0]),1):
                    plt.imshow(Cells[t,:,:]),plt.colorbar(),
                    plt.savefig(str(savepath+"Cells_{}.png".format(t)))
                    plt.close()
    
    return 