import numpy as np
from ModelFuncs import TransientFlowModel, vector_arrows
import matplotlib.pyplot as plt

# TODO: Add exponentially increasing k and density with vertical distance below surface

# time is measured in DAYS

length = 2000
width = 1000
depth = 2
cell_spacing_xy = 10
cell_spacing_z = 1
H_slope = 0.3 # topographic slope from upper to lower boundary, 1 = lose as much height as horizontal distance
H_base_elevation = 0 # raise entire surface this far above sea level
H_noise_multiplier = 0.1 # increase noise for initial head definition - when noise_multiplier=1 random noise is distributed between 0-1
loss_at_edges = 5 # extraction rate at glacier sides m3/d
loss_at_terminus = 5
moulin_location = None #((50,60),(50,60)) # give cell indices for horizontal extent in 1st tuple, vertical extent in 2nd tuple, or set to None
moulin_extr_rate = 200 #rate of extraction via moulin, m3/d
t = np.arange(0,10,0.05) # time to run model over in days
Ss = 0.01 # specific storage in each cell
stream_location = None #((0,-10),(20,30))
savepath = '/home/joe/Code/FlowModel/Outputs/'
x = np.arange(0,width,cell_spacing_xy)
y = np.arange(0,length,cell_spacing_xy)
z = np.arange(0,depth,cell_spacing_z)
SHP = (len(z)-1, len(y)-1, len(x)-1)

melt_rate = np.zeros(SHP)#np.random.rand(len(z)-1,len(y)-1,len(x)-1)/1 # melt rate is in m3 liquid water per cell per day (cell volume = cell_spacing_xy**2 * cell_spacing_z)

plot_types = ['Q','Phi'] # select what to plot, options are Q (net inflow to cells), Qs (water released from storage), Qx (flow across lateral cell boundaries)
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


HI = np.random.rand(len(z)-1,len(y)-1,len(x)-1) * H_noise_multiplier

# set a downslope gradient in initial heads and melt rate
for i in range(HI.shape[1]):
    HI[:,i,:]+= (HI.shape[1] - (i*H_slope))/100
    melt_rate[:,i,:] += (melt_rate.shape[1] + (i*H_slope))/100

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
HI += H_base_elevation

IBOUND = np.ones(SHP)
IBOUND[:, -1, :] = -1 # last row of model heads are prescribed (-1 head at base boundary)
IBOUND[:, 0, :] = 0 # these cells are inactive (top boundary)


Out = TransientFlowModel(x, y, z, t, kx, ky, kz, Ss, FQ, HI, IBOUND, epsilon=0.67)

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
                plt.imshow(Out.Q[i,-1, 5:-5, 5:-5] ,vmin=-1,vmax=1)
                plt.colorbar()
                plt.savefig(str(savepath+'Net_inflow_at_t{}.png'.format(i)))
                plt.close()

        if plot_type == 'Qs':
            
            for i in range(len(t)-1):
                plt.figure(figsize=(12,12))
                plt.title('Net flow out of cells in layer {}'.format(plot_layer))
                plt.imshow(Out.Qs[i,-1, 5:-5, 5:-5])
                plt.colorbar()
                plt.savefig(str(savepath+'Net_outflow_at_t{}.png'.format(i)))
                plt.close()

        if plot_type == 'Qx':
            
            for i in range(len(t)-1):
                plt.figure(figsize=(12,12))
                plt.title('Lateral flow into cells in layer {}'.format(plot_layer))
                plt.imshow(Out.Qx[i,-1, 5:-5, 5:-5])
                plt.colorbar()
                plt.savefig(str(savepath+'Net_lateral_flow_at_t{}.png'.format(i)))
                plt.close()

        if plot_type == 'Qy':
            
            for i in range(len(t)-1):
                plt.figure(figsize=(12,12))
                plt.title('Net longitudinal flow into cells in layer {}'.format(plot_layer))
                plt.imshow(Out.Qy[i,-1, 5:-5, 5:-5])
                plt.colorbar()
                plt.savefig(str(savepath+'Net_longitudinal_flow_at_t{}.png'.format(i)))
                plt.close()

        if plot_type =='Phi':

            for i in range(len(t)-1):
                plt.figure(figsize=(12,12))
                plt.title('Hydraulic head in cell centres in layer {}'.format(plot_layer))
                plt.imshow(Out.Phi[i,-1, 5:-5, 5:-5],vmin=-30,vmax=30)
                plt.colorbar()
                plt.savefig(str(savepath+'Hydraulic_Head_at_t{}.png'.format(i)))
                plt.close()

print(Out.Phi[0,0,:,:])