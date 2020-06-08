import numpy as np
import matplotlib.pyplot as plt

def vector_arrows(Out, x, y, z, plot_layer):
    
    """
    returns flow velocity vectors in each cell
    Parameters
    ----------
    `Out` : namedtuple containing arrays `Qx`, `Qy`, `Qz` as 
        returned by FlowModelFuncs.TransientFlowModel 
   
    `x` : ndarray, [m]
        Grid line coordinates of columns
    'y' : ndarray, [m]
        Grid line coordinates of rows
    `z` : ndaray [L] | int [-]
        If z == None, then iz must be given (default = 0)
        If z is an ndarray vector of floats 
            z will be interpreted as the elevations of uniform layers.
            iz will be ignored
        If z is a full 3D ndarray of floats
            z will be interpreted as the elevations of the tops and bottoms of all cells.
            iz will be ignored
    `iz` : int [-] 
        iz is ignored if z ~= None
        iz is the number of the layer for which the data are requested, and all output 
        arrays will be 2D for that layer.    
    
    Returns
    -------
    `Xm` : ndarray, shape: (Nz, Ny, Nx), [L]
        x-coordinates of cell centers
    `Ym` : ndarray, shape: (Nz, Ny, Nx), [L]
        y-coodinates of cell centers
    `ZM` : ndarray, shape: (Nz, Ny, Nx), [L]
        `z`-coordinates at cell centers
    `U` : ndarray, shape: (Nz, Ny, Nx), [L3/d]
        Flow in `x`-direction at cell centers
    `V` : ndarray, shape: (Nz, Ny, Nx), [L3/T]
        Flow in `y`-direction at cell centers
    `W` : ndarray, shape: (Nz, Ny, Nx), [L3/T]
        Flow in `z`-direction at cell centers.
    """
    # length of array in each dimension
    Ny = len(y)-1
    Nx = len(x)-1
    Nz = len(z)-1

    print(Nz)

    # coordinates of cell centres
    # (halfway between L and R edges)
    xm = 0.5 * (x[:-1] + x[1:])
    ym = 0.5 * (y[:-1] + y[1:])
    zm = 0.5 * (z[:-1] + z[1:])

    # create empty arrays for output
    U = np.zeros((len(Out.Qx[:,0,0,0]),len(Out.Qx[0,:,0,0]),len(Out.Qx[0,0,:,0]),len(Out.Qx[0,0,0,:])+1))    
    V = np.zeros((len(Out.Qy[:,0,0,0]),len(Out.Qy[0,:,0,0]),len(Out.Qy[0,0,:,0])+1,len(Out.Qy[0,0,0,:])))
    W = np.zeros((len(Out.Qz[:,0,0,0]),len(Out.Qz[0,:,0,0])+1,len(Out.Qz[0,0,:,0]),len(Out.Qz[0,0,0,:])))

    # create mesh
    X, Y, = np.meshgrid(xm, ym) # coordinates of cell centers
    Z = np.meshgrid(zm)

    # iterate through timesteps
    for t in range(len(Out.Qy[:,0,0,0])): # number of timesteps

        #grab relevant timestep from Out array
        Qx = Out.Qx[t,:,:,:]
        Qy = Out.Qy[t,:,:,:]
        Qz = Out.Qz[t,:,:,:]

        # Calculate flows at cell centers by interpolating between L and R faces
        Ut = np.concatenate((Qx[plot_layer, :, 0].reshape((1, Ny, 1)), \
                            0.5 * (Qx[plot_layer, :, :-1].reshape((1, Ny, Nx-2)) +\
                                Qx[plot_layer, :, 1: ].reshape((1, Ny, Nx-2))), \
                            Qx[plot_layer, :, -1].reshape((1, Ny, 1))), axis=2).reshape((Nx,Ny))

        Vt = np.concatenate((Qy[plot_layer, 0, :].reshape((1, 1, Nx)), \
                            0.5 * (Qy[plot_layer, :-1, :].reshape((1, Ny-2, Nx)) +\
                                Qy[plot_layer, 1:,  :].reshape((1, Ny-2, Nx))), \
                            Qy[plot_layer, -1, :].reshape((1, 1, Nx))), axis=1).reshape((Nx,Ny))

        # average flow across vertical cell to get z flow at cell centre
        QzTop = Qz[0:-1,:,:]
        QzBot = Qz[1:,:,:]
        Wt = (QzTop+QzBot)/2
        
        # add results to output arrays
        U[t,:,:,:] = Ut
        V[t,:,:,:] = Vt
        W[t,1:-1,:,:] = Wt

    return X,Y,Z,U,V,W



def particle_tracker(Out, U, V, W, t, cell0, cellG, cellD, SHP, cryoconite_locations):

    """
    Assume bacteria are <5um and therefore gravitational settling and physical
    filtration are negligible

    base condition is that initial concentration in cells/mL is scaled by flow
    in mL/t, giving flux of cells per timestep. However, this is modified by
    growth and decay and potentially adsorption at cryoconite layers (and ice?).

    normalise the amoutn of total flow going in lateral and longitudinal direction in each layer
    so that x% flows L/R and x% flows N/S. sum all vertical layers.

    select a direction according to max vector? Then send appropriate cell number
    to appropriate adjacent cell. Should be able to vectorise this operation.

    cell_cnc0 = 5000000
    cell_cnc1 = cell_cnc0 + netFlux + growth - death + adsorption + gravitation + pickup from conite
    + dropoff to conite

    """

    Cells = np.random.rand(len(t),SHP[0],SHP[1],SHP[2])*cell0
    CellD = np.zeros(shape=(SHP[0],SHP[1],SHP[2]))+cellD
    CellG = np.zeros(shape=(SHP[0],SHP[1],SHP[2]))+cellG
    CellG[-1,cryoconite_locations]=cellG*100

    # calculate divergence as dot product of vector components in x,y and z directions
    
    for t in np.arange(0,len(Out.Qz[:,0,0,0]),1):
        for layer in np.arange(0,len(Out.Qz[0,:,0,0]),1):
            
            divergence = (U[t,layer,:,:] + V[t,layer,:,:] + W[t,layer,:,:])
            
            # divergence gives net in/outflow in m3/t
            # cells/m3 = cells/mL *1000

            delC = (0-divergence) * ((1+CellG[layer,:,:]) - CellD[layer,:,:])
            Cells[t,layer,:,:] = Cells[t,layer,:,:] + delC*1000 
            Cells[Cells<0] = 0
    
    Cells = np.sum(Cells,axis=1)

    return Cells

