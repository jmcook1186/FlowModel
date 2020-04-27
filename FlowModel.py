import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from FlowFuncs import FlowFuncs


###########################
# SET BOUNDARY CONDITIONS #

# ztop, zbot and h are all meassured as elevation above sea level.

# define grid shape and size
len_x = 50
len_y = 50
cell_spacing = 1

h_boundary = 80 # head at top, bottom and lateral glacier boundaries
t_lim = 500 # total timesteps to run over
t_step = 1
t0 = 0 
slope = 0.5  # slope as fraction
WCthickness = 2 # WC thickness in meters
initial_WT = 0.3 # initial water table height as fraction of WC thickness
melt_rate = 0.0002 # m of head/timestep)
K_magnitude = 0.1 # order of magnitude of K values (starts as random numbers between 0-1, provide multiplier to get desired magnitude)


# call functions to build grids and populate with param values
ztop, zbot, h, K, qE, qW, qS, qN = FlowFuncs.make_empty_grids(len_x,len_y,cell_spacing)
ztop, zbot, del_z, h, K, melt = FlowFuncs.fill_param_grids(ztop, zbot, h, K, slope, WCthickness, initial_WT, h_boundary, melt_rate, K_magnitude)

# configure plots
cbar_max = ztop.max()
cbar_min = zbot.min() - 10
savepath1 = '/home/joe/Code/MicroMelt/Code/FlowModel/FigSTART.jpg'
savepath2 = '/home/joe/Code/MicroMelt/Code/FlowModel/FigEND.jpg'
FlowFuncs.plot_grid(h, cbar_max, cbar_min, savepath1) # initial plot at t=0

# print stats at t=0
print("max_h = ",h[1:len_x-1,1:len_y-1].max(), "min_h = ", h[1:len_x-1,1:len_y-1].min(), "mean_h = ", h[1:len_x-1,1:len_y-1].mean(), "std_h = ", h[1:len_x-1,1:len_y-1].std())

# open loop and run model
for t in np.arange(t0,t_lim,t_step):

    h = FlowFuncs.run_model(t0,t_lim,t_step,len_x,len_y, h, cell_spacing, K, qE, qW, qS, qN, melt, ztop, zbot, h_boundary)

    # add influx from melt
    h = h+melt

    # at each timestep impose boundary values and check h does not exceed top surface or drop below bottom surface
    h = np.where(h>ztop, ztop, h)
    h = np.where(h< zbot, zbot, h)
    h[[0,-1],:] = h_boundary
    h[:,[0,-1]] = h_boundary

# plot at t= t_lim
FlowFuncs.plot_grid(h, cbar_max, cbar_min, savepath2)

# print stats at t = t_lim  
print("max_h = ",h[1:len_x-1,1:len_y-1].max(), "min_h = ", h[1:len_x-1,1:len_y-1].min(), "mean_h = ", h[1:len_x-1,1:len_y-1].mean(), "std_h = ", h[1:len_x-1,1:len_y-1].std())
