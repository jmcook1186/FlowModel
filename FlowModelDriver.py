import numpy as np
from FlowModelFuncs import TransientFlowModel
from ParticleModelFuncs import vector_arrows
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from BuildGlacier import Glacier

# TODO: Add exponentially increasing k and density with vertical distance below surface
# TODO: calculate volume of water in each cell from Q values, derive flow velocity and volume lost to extraglacial env

# time is measured in DAYS
# since we are simulating an unconfined aquifer, the hydraulic head is equal to the 
# water table height abov sea level

"""
Hydraulic head calculations:

Initial hydraulic head is calculated from elevation above sea level of the weathring crust lower boundary
and the initial water table height above the lower surface. This gives the hydraulic head at the water table,
but we calculate head and flow at a given resolution within the water table, so the head must be calculated
for each vertical step as defined by the WC_thickness and cell_spacing_z (so for a 5m thick WC and a vertical
resolution of 1m, 5 hydraulic head values are required). This is achieved using the equation:

#     h = psi + z 
# where h = hydraulic head, psi = pressure head (i.e. elevation difference between measurement point 
# and water table, and z = elevation at the measurement point). These calculated hydraulic heads are 
# added to the appropriate layer of HI and the initial heads are thus defined.

"""

# program config
savepath = '/home/joe/Code/FlowModel/Outputs/'
epsilon = 0.67
plot_types = ['Q', 'Qy', 'Phi3D'] # select what to plot, options are Q (net inflow to cells), Qs (water released from storage), Qx (flow across lateral cell boundaries)
#Qy (flow across longitudinal cell boundaries), Phi (hydraulic head at cell centres). Provide as list of strings or set to "None".
plot_layer = 0 # which vertical layer to plot (0 = top, -1 = bottom)
figsize = (15,15)
t = np.arange(0,1,0.05) # time to run model over in days

# grid size
length = 100
width = 100
WC_thickness0 = 3 # initial WC thickness at t=0
cell_spacing_xy = 1 # size of cells in meters in horizontal dimension
cell_spacing_z = 1 # size of cells in meters in vertical dimension
x = np.arange(0,width,cell_spacing_xy)
y = np.arange(0,length,cell_spacing_xy)
z = np.arange(0,WC_thickness0,cell_spacing_z)
kxy = 3.15
kz = 1.15

# environmental variables
slope = 1.0 # topographic slope from upper to lower boundary, 1 = lose as much height as horizontal distance
base_elevation = 100 # raise entire surface this far above sea level
WaterTable0 = 0.3 # proportion of WC filled with water at t=0
melt_rate0 = 0.1 # water added by melting in m3/d
rainfall0 = 0 # water added by rainfall in m3/d
loss_at_edges = 200 # extraction rate at glacier sides m3/d
loss_at_terminus = 2000  # extraction rate at glacier termins
cryoconite_coverage = 0.05 # fraction of total surface covered by cryoconite holes
moulin_location = None #((50,60),(50,60)) # give cell indices for horizontal extent in 1st tuple, vertical extent in 2nd tuple, or set to None
moulin_extr_rate = 200 #rate of extraction via moulin, m3/d
stream_location = None #((0,-10),(20,30))
constrain_head_to_WC = True # toggling this ON means the hydraulic head cannot rise above the upper glacier surface nor drop below the lower WC boundary
Ss = 0.01 # specific storage in each cell

# BUILD GLACIER
glacier = Glacier(x, y, z, cell_spacing_xy, cell_spacing_z, base_elevation, WC_thickness0, WaterTable0, \
    cryoconite_coverage, melt_rate0, rainfall0, slope, kxy, kz, loss_at_edges, loss_at_terminus,\
    stream_location, moulin_location, moulin_extr_rate)

Out = TransientFlowModel(x, y, z, t, glacier.kx, glacier.ky, glacier.kz, Ss, glacier.FQ,\
     glacier.HI, glacier.IBOUND, epsilon, glacier.upper_surface, glacier.lower_surface, 
     constrain_head_to_WC, moulin_location)



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


# calculate flow vectors between cells
X,Y,U,V = vector_arrows(Out, x, y, z, plot_layer)

color_array = np.arange(0,100,1)

for t in np.arange(0,len(U[:,0,0]),1):
    
    Ut = U[t,1,:,:]
    Vt = V[t,1,:,:]
    plt.figure(figsize=figsize)
    plt.quiver(X,Y,Ut,Vt,color_array, cmap='autumn',scale=10000)
    plt.savefig('/home/joe/Code/FlowModel/VectorFig{}.png'.format(t))
    plt.close()
