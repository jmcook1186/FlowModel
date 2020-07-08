import numpy as np
from FlowModelFuncs import TransientFlowModel
from ParticleModelFuncs import vector_arrows, cell_cnc_tracker, calc_export
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from BuildGlacier import Glacier
from PlottingTools import plotFigures

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
plot_types = ["Q","Phi","Cell Export at Terminus", "Cumulative Cell Export at Terminus"] # select what to plot, options are Q (net inflow to cells), Qs (water released from storage), Qx (flow across lateral cell boundaries)
#Qy (flow across longitudinal cell boundaries), Phi (hydraulic head at cell centres). Provide as list of strings or set to "None".
plot_layer = 0 # which vertical layer to plot (0 = top, -1 = bottom)
figsize = (15,15)
t = np.arange(0,1,0.01) # time to run model over in days
lat = 67.04 # site latitude
lon = 49.99 # site longitude
day = 202 # day of year
time = 1500 # time of day (24hr)

# grid size
length = 200
width = 100
WC_thickness0 = 5 # initial WC thickness at t=0
cell_spacing_xy = 1 # size of cells in meters in horizontal dimension
cell_spacing_z = 1 # size of cells in meters in vertical dimension
x = np.arange(0, width, cell_spacing_xy)
y = np.arange(0, length, cell_spacing_xy)
z = np.arange(0, WC_thickness0, cell_spacing_z)
kxy = 3.15
kz = 0.75

# environmental variables
slope = 5 # topographic slope from upper to lower boundary, 1 = lose as much height as horizontal distance
aspect = 180 # degrees, N is 0
roughness = 0.005 # default is 0.005
base_elevation = 100 # raise entire surface this far above sea level
WaterTable0 = 0.3 # proportion of WC filled with water at t=0
melt_rate0 = 0.00001 # water added by melting in m3/d
rainfall0 = 0 # water added by rainfall in m3/d
loss_at_edges = 2 # extraction rate at glacier sides m3/d
loss_at_terminus = 200  # extraction rate at glacier termins
cryoconite_coverage = 0.05 # fraction of total surface covered by cryoconite holes
moulin_location = None #((50,60),(50,60)) # give cell indices for horizontal extent in 1st tuple, vertical extent in 2nd tuple, or set to None
moulin_extr_rate = 200 #rate of extraction via moulin, m3/d
stream_location = None #((0,-10),(20,30))
constrain_head_to_WC = True # toggling this ON means the hydraulic head cannot rise above the upper glacier surface nor drop below the lower WC boundary
porosity0 = 0.3 # initial porosity in each cell
specific_retention = 0.05 # tune-able, describes proportion of water left behind after aquifer drains - set to value for fine gravel from Bear et al (1973)
MELT_CALCS = True # toggle whether to initiate the albedo-melt-porosity feedback calculations
algae = 0 # select from discrete "levels" of surface glacier algal loading between 0 (clean) and 3 (high concentration)

# meteorological variables
lapse = 0.65 
windspd = 1.5 
airtemp = 0.01
inswrd = 55 
avp = 900 

#microbial variables
cell0 = 10000 # initial cell concentration in cell/mL
cellG = 0.72 # growth rate
cellD = 0.35 # death rate

# BUILD GLACIER
glacier = Glacier(x, y, z, cell_spacing_xy, cell_spacing_z, base_elevation, WC_thickness0, porosity0, specific_retention, WaterTable0, \
    cryoconite_coverage, melt_rate0, rainfall0, slope, kxy, kz, loss_at_edges, loss_at_terminus,\
    stream_location, moulin_location, moulin_extr_rate, algae, lat, lon, day, time, aspect,\
        roughness, lapse, windspd, airtemp, inswrd, avp)

# CALCULATE FLOWS
Out,porosity = TransientFlowModel(x, y, z, t, glacier, epsilon, constrain_head_to_WC, MELT_CALCS, moulin_location)

# CALCULATE COMPONENT VECTORS
X,Y,Z,U,V,W = vector_arrows(Out, x, y, z, plot_layer)

# CALCULATE CELL FLUXES
Cells, CellColumnTot = cell_cnc_tracker(Out, U, V, W, t, cell0, cellG, cellD, glacier.SHP, glacier.cryoconite_locations)

# SUMMARISE CELL FLUXES
TotalExport, CumExport = calc_export(CellColumnTot)

# PLOT FIGURES
plotFigures(x, y, z, plot_types, Out, t, plot_layer, Cells, CellColumnTot, TotalExport, CumExport, figsize, savepath)