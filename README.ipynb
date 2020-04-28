# FlowModel
Model subsurface flow through a glacier weathering crust

## Background

This code simulates the three dimensional flow of water through a glacier weathering crust. To achieve this, water is assumed to obey Darcy's Law of flow through porous media on the assumption that water is incompressible, laminar and has Reynold's number < 1. Darcy's Law relates the specific discharge vector (Q), the hydraulic conductivity (K) and the hydraulic head (h). Flow occurs from regions of higher head to regions of lower head. This flow model takes its inspiration from the groundwater flow module in the USGS hydrological model MODFLOW.

The 3D flow of groundwater through the weathering crust can be described by the partial differential equation:

$$ \frac{\delta}{\delta x} (K . \frac{\delta h}{\delta x} ) + 
\frac{\delta}{\delta y} (K . \frac{\delta h}{\delta y} ) +
\Upsilon = 0 $$

(Equation 1)

where 
$K$ = hydraulic conductivity, $x$ and $y$ = cartesian coordinates, $\Epsilon$$ is a source term representing distributed inputs or outputs of water. In this case, the source term $\Upsilon$ represents water derived from ice melting, which is distributed across the model area. The numerical solution to this equation is found using the finite difference method where the continuous system is discretized by dividing it into a set of discrete cells distributed evenly over a finite grid. An initial glacier surface ($z_{top}$) is defined given a slope along the long axis of the glacier plus some random noise. The weathering crust is defined by creating a second surface ($z_{bot}$) a given distance ($del_{z}$) beneath the upper surface. The water table thickness is defined by assigning a proportion of del_z that is saturated, providing initial values for hydraulic head ($h$). Note that the upper surface, lower surface and hydraulic head are all measured in metres elevation above sea level.

Saturated thickness (SatT) is calculated as:

$$ \delta_z  (if h > z_{top}) $$
$$ h - z_{bot}   (if z_{bot} < h < z_{top}) $$
$$ 0 (if h < z_{bot}) $$            
(Equation 2)

such that if the hydraulic head exceeds the upper surface, the saturated thickness is equal to the entire weathering crust thickness. If the hydraulic head is below the lower surface the saturation thickness is 0. Otherwise, the saurated thickness is the elevation of the bottom surface subtracted from the elevation of the hydraluic head.

The discretized version of equation 1 is as follows:



$${K}_{mean}  * \frac{{h_{i-1},_j}-{h_i,_j}} {\Delta x} \Delta y \Upsilon _{mean}  + 
{K}_{mean}  * \frac{{h_{i+1},_j}-{h_i,_j}} {\Delta x} \Delta y \Upsilon _{mean}  +
{K}_{mean}  * \frac{{h_i,_{j-1}}-{h_i,_j}} {\Delta x} \Delta y \Upsilon _{mean}  +
{K}_{mean}  * \frac{{h_i,_{j+1}}-{h_i,_j}} {\Delta x} \Delta y \Upsilon _{mean}  +
\Psi_{i,j}
$$ 

(Equation 3)

where $K_{mean}$ is the arithmetic mean of the hydraulic conductivity of the central cell and a neighbour. $\Upsilon_{mean}$ is the arithmetic mean of the saturated thickness of the cell and its neighbour, and $\Psi$ is a source term representing water arriving in the cell due to in situ melting. Equation 3 gives the total flux of water into the cell, where positive values indicate flow into the cell and negative vaues indicate flow out of the cell. 

The hydraulic head in cell[i,j] is then equal to:

$$ h_{i,j} = h_{i,j} + FluxTotal $$

(Equation 4)

The boundaries of the grid are reset in each timestep to a constant value (h_boundary) and omitted from analysis to prevent edge effects rippling into the main grid. At the end of each timestep:

$$ h_{[[0:-1],:]} = h_{boundary} $$
$$ h_{[:,[0,-1]]} = h_{boundary} $$

(Equation 5)


## Outputs
The following figures show the hydraulic head in the weathering crust (blue) with the upper glacier surface overlain on top (grey) at t = 0 and t = t_lim (in this case t_lim = 100 timesteps).

![t=0](/Assets/FigSTART.png "Hydraulic heads at t=0")
![t=500](/Assets/FigEND.png "Hydraulic heads at t=500")

## How to Use

The program is controlled via FlowModel.py and the relevat functions are contained in FlowFuncs.py

## Permissions

This software is in active development and no permissions are granted for any other use. This software exists without any warranty nor implied warranty for any purpose.