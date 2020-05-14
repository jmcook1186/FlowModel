import numpy as np
from ModelFuncs import TransientFlowModel, vector_arrows
import matplotlib.pyplot as plt

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


savepath = '/home/joe/Code/FlowModel/Outputs/'
epsilon = 0.67
length = 1000
width = 800
WC_thickness0 = 3 # initial WC thickness at t=0
cell_spacing_xy = 10
cell_spacing_z = 1
slope = 0.3 # topographic slope from upper to lower boundary, 1 = lose as much height as horizontal distance
base_elevation = 100 # raise entire surface this far above sea level
WaterTable0 = 0.3 # proportion of WC filled with water at t=0

loss_at_edges = 5 # extraction rate at glacier sides m3/d
loss_at_terminus = 5
moulin_location = None #((50,60),(50,60)) # give cell indices for horizontal extent in 1st tuple, vertical extent in 2nd tuple, or set to None
moulin_extr_rate = 200 #rate of extraction via moulin, m3/d

t = np.arange(0,5,0.05) # time to run model over in days
Ss = 0.01 # specific storage in each cell

stream_location = None #((0,-10),(20,30))

x = np.arange(0,width,cell_spacing_xy)
y = np.arange(0,length,cell_spacing_xy)
z = np.arange(0,WC_thickness0,cell_spacing_z)
SHP = (len(z)-1, len(y)-1, len(x)-1)

upper_surface = (np.zeros(SHP[1:]) + base_elevation)  + np.random.rand(SHP[1],SHP[2])/10
lower_surface = (upper_surface - WC_thickness0) + np.random.rand(SHP[1],SHP[2])/10
WaterTable0 = lower_surface + (WC_thickness0*WaterTable0)
melt_rate0 = np.zeros(SHP)+0.02

for i in range(upper_surface.shape[0]):

    upper_surface[i,:] += (upper_surface.shape[0] - (i*slope))/10
    lower_surface[i,:] += (lower_surface.shape[0] - (i*slope))/10
    WaterTable0[i,:] += (WaterTable0.shape[0] - (i*slope))/10
    
#3D water table
HI = np.zeros(SHP)
HI[0,:,:] = WaterTable0

# calculate hydraulic head at each finite difference length beneath the water tbale surface
for i in np.arange(1,WC_thickness0/cell_spacing_z-1,1):
    HI[int(i),:,:] = WaterTable0 - cell_spacing_z + (cell_spacing_z*i)


melt_rate = np.zeros(SHP)#np.random.rand(len(z)-1,len(y)-1,len(x)-1)/1 # melt rate is in m3 liquid water per cell per day (cell volume = cell_spacing_xy**2 * cell_spacing_z)

plot_types = ['Qy','Phi3D'] # select what to plot, options are Q (net inflow to cells), Qs (water released from storage), Qx (flow across lateral cell boundaries)
                    #Qy (flow across longitudinal cell boundaries), Phi (hydraulic head at cell centres). Provide as list of strings or set to "None".
plot_layer = 0 # which vertical layer to plot (0 = top, -1 = bottom)


# set hydraulic conductivity in m/d - these are 3D arrays where the conductivity through the horizontal (x), 
# horizontal (y) or vertical (z) faces are defined - glacier average from Stevens et al (2018) = 0.185
kx = (np.random.rand(len(z)-1,len(y)-1,len(x)-1)/10)+3.85 # [m/d] 3D kx array
ky = (np.random.rand(len(z)-1,len(y)-1,len(x)-1)/10)+3.85 # [m/d] 3D ky array with same values as kx
kz = (np.random.rand(len(z)-1,len(y)-1,len(x)-1)/10)+3.85 # [m/d] 3D kz array with same values as kx


FQ = np.zeros(SHP) # all flows zero. Note sz is the shape of the model grid
FQ[:, :, [0,-1]] = -loss_at_edges # [m3/d] extraction in these cells - drawdown at side boundaries
FQ[:,-1,:] = -loss_at_terminus


#set initial values to zero within moulin

if moulin_location != None:

    FQ[:,moulin_location[0][0]:moulin_location[0][1],moulin_location[1][0]:moulin_location[1][1]] = -moulin_extr_rate
    HI[:,moulin_location[0][0]:moulin_location[0][1],moulin_location[1][0]:moulin_location[1][1]] = -1000
    melt_rate[:,moulin_location[0][0]:moulin_location[0][1],moulin_location[1][0]:moulin_location[1][1]] = 0

if stream_location != None:
    kx[:,stream_location[0][0]:stream_location[0][1],stream_location[1][0]:stream_location[1][1]] = 10
    ky[:,stream_location[0][0]:stream_location[0][1],stream_location[1][0]:stream_location[1][1]] = 10
    kz[:,stream_location[0][0]:stream_location[0][1],stream_location[1][0]:stream_location[1][1]] = 10
    FQ[:,stream_location[0][0]:stream_location[0][1],stream_location[1][0]:stream_location[1][1]] = 0
    HI[:,stream_location[0][0]:stream_location[0][1],stream_location[1][0]:stream_location[1][1]] = -1
    melt_rate[:,stream_location[0][0]:stream_location[0][1],stream_location[1][0]:stream_location[1][1]] = 0

FQ += melt_rate

IBOUND = np.ones(SHP)
IBOUND[:, -1, :] = -1 # last row of model heads are prescribed (-1 head at base boundary)
IBOUND[:, 0, :] = 0 # these cells are inactive (top boundary)


Out = TransientFlowModel(x, y, z, t, kx, ky, kz, Ss, FQ, HI, IBOUND, epsilon)

print('Out.Phi.shape ={0}'.format(Out.Phi.shape))
print('Out.Q.shape ={0}'.format(Out.Q.shape))
print('Out.Qx.shape ={0}'.format(Out.Qx.shape))
print('Out.Qy.shape ={0}'.format(Out.Qy.shape))
print('Out.Qz.shape ={0}'.format(Out.Qz.shape))


if plot_types != None:

    for plot_type in plot_types:
        
        if plot_type == 'Q':

            for i in range(len(t)-1):

                plt.figure(figsize=(12,12))
                plt.title('Net flow into cells in layer {}'.format(plot_layer))
                plt.imshow(Out.Q[i,plot_layer, 5:-5, 5:-5],vmin=-5,vmax=5)
                plt.colorbar()
                plt.savefig(str(savepath+'Net_inflow_at_t{}.png'.format(i)))
                plt.close()

        if plot_type == 'Qs':
            
            for i in range(len(t)-1):
                plt.figure(figsize=(12,12))
                plt.title('Net flow out of cells in layer {}'.format(plot_layer))
                plt.imshow(Out.Qs[i,plot_layer, 5:-5, 5:-5])
                plt.colorbar()
                plt.savefig(str(savepath+'Net_outflow_at_t{}.png'.format(i)))
                plt.close()

        if plot_type == 'Qx':
            
            for i in range(len(t)-1):
                plt.figure(figsize=(12,12))
                plt.title('Lateral flow into cells in layer {}'.format(plot_layer))
                plt.imshow(Out.Qx[i,plot_layer, 5:-5, 5:-5])
                plt.colorbar()
                plt.savefig(str(savepath+'Net_lateral_flow_at_t{}.png'.format(i)))
                plt.close()

        if plot_type == 'Qy':
            
            for i in range(len(t)-1):
                plt.figure(figsize=(12,12))
                plt.title('Net longitudinal flow into cells in layer {}'.format(plot_layer))
                plt.imshow(Out.Qy[i,plot_layer, 5:-5, 5:-5],vmin=-20,vmax=20)
                plt.colorbar()
                plt.savefig(str(savepath+'Net_longitudinal_flow_at_t{}.png'.format(i)))
                plt.close()

        if plot_type =='Phi':

            for i in range(len(t)-1):
                plt.figure(figsize=(12,12))
                plt.title('Hydraulic head in cell centres in layer {}'.format(plot_layer))
                plt.imshow(Out.Phi[i,plot_layer, 5:-5, 5:-5],vmin=0,vmax=110)
                plt.colorbar()
                plt.savefig(str(savepath+'Hydraulic_Head_at_t{}.png'.format(i)))
                plt.close()

        if plot_type == 'Phi3D':

            for i in range(len(t)-1):
                X,Y = np.meshgrid(x[6:-5],y[6:-5])
                Z = Out.Phi[i,plot_layer,5:-5,5:-5]

                ax = plt.axes(projection='3d')
                ax.plot_surface(Y, X, Z, cmap='winter', edgecolor='none')
                ax.set_title('Hydraulic Head at t{}'.format(i))
                ax.set_zlim(90,110)
                plt.savefig(str(savepath+'Hydraulic_Head_at_t{}.png'.format(i)))
                plt.close()

print(Out.Phi[0,0,:,:])