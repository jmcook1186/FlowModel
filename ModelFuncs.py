import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from collections import namedtuple


def vector_arrows(Out, x, y, z=None, iz=0):
    
    """
    returns coordinates and velocity components to show velocity with quiver

    Compute arrays to display velocity vectors using mpl quiver
    Quiver is always drawn in the xy plane ofr specific layer
    default iz=0

    Parameters
    ----------
    `Out` : namedtuple holding arrays `Qx`, `Qy`, `Qz` as defined in `fdm3`
        `Qx` : ndarray, shape: (Ny, Nx-1, Nz), [L3/T]
            Interfacial flows in finite difference model in x-direction from `fdm3'
        `Qy` : ndarray, shape: (Ny-1, Nx, Nz), [L3/T]
            Interfacial flows in finite difference model in y-direction from `fdm3`
        `Qz` : ndarray, shape: (Ny, Nx, Nz-1), [L3/T]
            Interfacial flows in finite difference model in z-direction from `fdm3`
    `x` : ndarray, [m]
        Grid line coordinates of columns
    'y' : ndarray, [m]
        Grid line coordinates of rows
    `z` : ndaray [L] | int [-]
        If z == None, then iz must be given (default = 0)
        If z is an ndarray vector of floats 
            z will be interpreted as the elvations of uniform layers.
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

    Ny = len(y)-1
    Nx = len(x)-1
    xm = 0.5* (x[:-1] + x[1:])
    ym = 0.5* (x[:-1] + x[1:])

    X,Y = np.meshgrid(xm,ym) #coordinates of cell centres

    #Flows at cell centres
    U = np.concatenate((Out.Qx[iz, :, 0].reshape((1, Ny, 1)), 0.5*(Out.Qx[iz, :, :-1].reshape((1, Ny, Nx-2)) + Out.Qx[iz, :, 1: ].reshape((1, Ny, Nx-2))), Out.Qx[iz, :, -1].reshape((1, Ny, 1))), axis=2).reshape((Ny,Nx))
    
    
    V = np.concatenate((Out.Qy[iz, 0, :].reshape((1, 1, Nx)), 0.5*(Out.Qy[iz, :-1, :].reshape((1, Ny-2, Nx)) + Out.Qy[iz, 1:,  :].reshape((1, Ny-2, Nx))), Out.Qy[iz, -1, :].reshape((1, 1, Nx))), axis=1).reshape((Ny,Nx))

    return X,Y,U,V


def sort_dim(x, tol=0.0001):

    """
    returns sorted unique values of x, keeping ascending or descending direction
    """

    if x[0] > x[-1]: # if array reversed
        x = np.sort(x)[::-1] # sort and reverse
        return x[np.hstack((np.diff(x) < -tol, True))]
    else:
        x = np.sort(x)
        return x[np.hstack((np.diff(x) > +tol, True))]


def TransientFlowModel(x, y, z, t, kx, ky, kz, Ss, FQ, HI, IBOUND, epsilon=0.67):
    
    '''Returns computed heads of steady state 3D finite difference grid.Steady state 3D Finite Difference Model 
    that computes the heads a 3D ndarray.
    
    Parameters
    ----------
    `x` : ndarray, shape: Nx+1, [L]
        `x` coordinates of grid lines perpendicular to rows, len is Nx+1
    `y` : ndarray, shape: Ny+1, [L]
        `y` coordinates of grid lines along perpendicular to columns, len is Ny+1
    `z` : ndarray, shape: Nz+1, [L]
        `z` coordinates of layers tops and bottoms, len = Nz+1
    t : ndarray, shape: [Nt+1]
        times at which the heads and flows are desired including the start time,
        which is usually zero, but can have any value.
    `kx`, `ky`, `kz` : ndarray, shape: (Ny, Nx, Nz) [L/T]
        hydraulic conductivities along the three axes, 3D arrays.
    `FQ` : ndarray, shape: (Ny, Nx, Nz), [L3/T]
        prescribed cell flows (injection positive, zero of no inflow/outflow)
    `IH` : ndarray, shape: (Ny, Nx, Nz), [L]
        initial heads. `IH` has the prescribed heads for the cells with prescribed head.
    `IBOUND` : ndarray of int, shape: (Ny, Nx, Nz), dim: [-]
        boundary array like in MODFLOW with values denoting
        *IBOUND>0  the head in the corresponding cells will be computed
        *IBOUND=0  cells are inactive, will be given value NaN
        *IBOUND<0  coresponding cells have prescribed head
    `epsilon` : float, dimension [-]degree of implicitness, choose value between 0.5 and 1.0

    Returns
    -------

    `Out` : namedtuple containing heads and flows:
        `Out.Phi` : ndarray, shape: (Ny, Nx, Nz), [L3/T] 
            computed heads. Inactive cells will have NaNs
            To get heads at time t[i], use Out.Phi[i]
            Out.Phi[0] = initial heads
        `Out.Q`   : ndarray, shape: (Ny, Nx, Nz), [L3/T]
            net inflow in all cells, inactive cells have 0
        `Out.Qs`  : ndarray, shape: (Nt, Nz, Ny, Nx), [L3/T]
            release from storage during time step.
        `Out.Qx   : ndarray, shape: (Ny, Nx-1, Nz), [L3/T]
            intercell flows in x-direction (parallel to the rows)
        `Out.Qy`  : ndarray, shape: (Ny-1, Nx, Nz), [L3/T]
            intercell flows in y-direction (parallel to the columns)
        `Out.Qz`  : ndarray, shape: (Ny, Nx, Nz-1), [L3/T]
            intercell flows in z-direction (vertically upward postitive)
            the 3D array with the final heads with `NaN` at inactive cells.

    '''

    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    import matplotlib.pyplot as plt

    Out = namedtuple('Out',['t','Phi', 'Q','Qs', 'Qx', 'Qy', 'Qz'])

    # use the sort_dim() function to ensure directionality is correct in each dimension
    x = sort_dim(x)
    y = sort_dim(y)[::-1]  # unique and descending
    z = sort_dim(z)[::-1]  # unique and descending

    # determine shape of array from lengths of each dimension
    SHP = Nz, Ny, Nx = len(z) - 1, len(y)-1, len(x)-1
    Nod = np.prod(SHP) # Nod is the total number of cells in the gird (x.y.z)

    # Nod = 0 when length of any dimension = 0
    if Nod == 0:
        raise AssertationError("Nx, Ny and Nz must be >= 1")

    # array where each element is the step size between array values,
    # reshaped into vectors
    # e.g. for array [0,2,4,6], diff = [2,2,2,2]
    dx = np.abs(np.diff(x).reshape(1, 1, Nx))
    dy = np.abs(np.diff(y).reshape(1, Ny,1))
    dz = np.abs((np.diff(z)).reshape(Nz, 1,1))

    # create boolean arrays for active, inactive and fixed-head cells
    active = (IBOUND>0).reshape(Nod) #vector of booleans
    inact  = (IBOUND==0).reshape(Nod) # dito for inact
    fxhd   = (IBOUND<0).reshape(Nod) # dito for fxhd

    # calculate inter-cell resistances
    # Rx is the resistance across the horizontal cell faces aligned in the x direction
    # Ry is the resistance across the horizontal cell faces aligned in the y direction
    # Rz is the resistance across the vertical cell faces
    Rx1 = 0.5*dx / (dy*dz) / kx
    Rx2 = Rx1
    Ry1 = 0.5*dy / (dz*dx) / ky
    Ry2 = Ry1
    Rz1 = 0.5*dz / (dx*dy) / kz
    Rz2 = Rz1

    # make inactive cells infinite resistance
    Rx1[inact.reshape(SHP)] = np.inf
    Rx2[inact.reshape(SHP)] = np.inf
    Ry1[inact.reshape(SHP)] = np.inf
    Rz1[inact.reshape(SHP)] = np.inf
    
    # conductances between adjacent cells
    # size of the arrays is dimension length - 2 because
    # in each dimension there are faces that do not have neighboursat either edge
    # this will be replicated in the cell number arrays IE,IW,IS...

    Cx = 1 / (Rx1[  :, :,  :-1] + Rx2[:, : , 1:])
    Cy = 1 / (Ry1[  :, :-1,:  ] + Ry2[:, 1:, : ])
    Cz = 1 / (Rz1[:-1, :,  :  ] + Rz2[1:, :, : ])

    # storage term, variable dt not included
    Cs = (Ss*(dx*dy*dz) / epsilon).ravel()

    # NOD reshapes Nod to SHP such that the  
    NOD = np.arange(Nod).reshape(SHP)

    # grab cell numbers for neighbours on each side
    # each of these arrays are shifted one element in the appropriate direction
    IE = NOD[:, :, 1:] # numbers of the eastern neighbors of each cell
    IW = NOD[:, :, :-1] # same western neighbors
    IN = NOD[:, :-1,:] # same northern neighbors
    IS = NOD[:, 1:,:] # southern neighbors
    IT = NOD[:-1,:,:] # top neighbors
    IB = NOD[1:,:,:] # bottom neighbors

    R = lambda x : x.ravel() # define lamdba function for ravelling arrays into vectors,
    #just to make the following expression more readable

    # Here, The arrays of conductances and resistances for x, y and z oriented faces are
    # first ravelled and then concatenated along common dimensions (i.e. a matrix is 
    # generated where each column contains the values for R or C for x,y or z orientation).
    # The same is then applied twice to the numbers of cells in each direction, first
    # creating an array ordered E,W,N,S,B,T then reveersing each pair: W,E,S,N,T,B. The 
    # final tuple contains the row and column indexes, Nod. The result is a 3-D sparse matrix
    # where each element has a conductance and the indexes of the cell either side of it in
    # each dimension.

    # scipy's compressed sparse column matrix (csc) is used to enable memory efficient storage
    # of the arrays and to format for linear algebraic operations, optimised for column-wise
    # operations.

    A = sp.csc_matrix(( np.concatenate(( R(Cx), R(Cx), R(Cy), R(Cy), R(Cz), R(Cz)) ),
    (np.concatenate(( R(IE), R(IW), R(IN), R(IS), R(IB), R(IT)) ),\
        np.concatenate(( R(IW), R(IE), R(IS), R(IN), R(IT), R(IB)) ))),(Nod,Nod))

    # the sign is reversed and the coefficient vector representing Ss(V/epsilon*delta_t)*ht
    # (i.e. water released from storage over timestep epsilon*delta_t) is added to the 
    # matrix primary diagonal (i.e. to those elements where the C value matches up with
    # an appropriate cell number for a neighbour in the correct direction - where Cx (left)
    # coincides with index of neighbour adjacent to the left)

    A = -A + sp.diags(np.array(A.sum(axis=1))[:,0])
    
    # number of timesteps is one less than length t due to zero-indexing
    Nt = len(t)-1

    # set up output arrays
    Out.Phi = np.zeros((Nt+1, Nod)) # Nt+1 times
    Out.Q   = np.zeros((Nt  , Nod)) # Nt time steps
    Out.Qs  = np.zeros((Nt  , Nod))
    Out.Qx  = np.zeros((Nt, Nz, Ny, Nx-1))
    Out.Qy  = np.zeros((Nt, Nz, Ny-1, Nx))
    Out.Qz  = np.zeros((Nt, Nz-1, Ny, Nx))

    # reshape input arrays to vectors using our ravel shorthand R
    # to enable vector multiplication with system matrix A

    FQ = R(FQ);  HI = R(HI);  Cs = R(Cs)

    # initialize heads
    Out.Phi[0] = HI

    # solve heads at active locations at t_i+eps*dt_i
    Nt=len(t)  # for heads, at all times Phi at t[0] = initial head

    Ndt=len(np.diff(t)) # for flows, average within time step

    for idt, dt in enumerate(np.diff(t)):

        it = idt + 1

        # compute right hand side of equation
        RHS = FQ - (A + sp.diags(Cs / dt))[:,fxhd].dot(Out.Phi[it-1][fxhd]) 

        # use RHS (computed from knowns) to solve matrix computation for next time step
        Out.Phi[it][active] = spsolve( (A + sp.diags(Cs / dt))[active][:,active], RHS[active] + Cs[active] / dt*Out.Phi[it-1][active])

        # calculate net flow into cell
        Out.Q[idt]  = A.dot(Out.Phi[it])

        # calculate net flow out of cell
        Out.Qs[idt] = -Cs/dt*(Out.Phi[it]-Out.Phi[it-1])

        # calculate flows across cell faces in each dimension
        Out.Qx[idt] =  -np.diff( Out.Phi[it].reshape(SHP), axis=2)*Cx
        Out.Qy[idt] =  +np.diff( Out.Phi[it].reshape(SHP), axis=1)*Cy
        Out.Qz[idt] =  +np.diff( Out.Phi[it].reshape(SHP), axis=0)*Cz

        # update head to end of time step
        Out.Phi[it] = Out.Phi[it-1] + (Out.Phi[it] - Out.Phi[it-1]) / epsilon

    # reshape Phi to shape of grid
    Out.Phi = Out.Phi.reshape((Nt,) + SHP)
    Out.Q   = Out.Q.reshape( (Ndt,) + SHP)
    Out.Qs  = Out.Qs.reshape((Ndt,) + SHP)

    return Out # all outputs in a named tuple for easy access