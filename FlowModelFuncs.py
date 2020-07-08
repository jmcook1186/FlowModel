import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from collections import namedtuple
import ebmodel as ebm
from math import factorial
import dask


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



def TransientFlowModel(x, y, z, t, glacier, epsilon, constrain_head_to_WC, MELT_CALCS, moulin_location):
    
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
    `kx`, `ky`, `kz` : ndarray, shape: (Ny, Nx, Nz) [m/d]
        hydraulic conductivities along the three axes, 3D arrays.
    `FQ` : ndarray, shape: (Ny, Nx, Nz), [L3/T]
        prescribed cell flows (injection positive, zero of no inflow/outflow)
    `HI` : ndarray, shape: (Ny, Nx, Nz), [L]
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

    Out = namedtuple('Out',[ 't', 'Phi', 'Q', 'Qs', 'Qx', 'Qy', 'Qz' , 'BBA', 'melt'])

    # use the sort_dim() function to ensure directionality is correct in each dimension
    x = sort_dim(x)
    y = sort_dim(y)[::-1]  # unique and descending
    z = sort_dim(z)[::-1]  # unique and descending

    SHP = glacier.SHP
    kx = glacier.kx
    ky = glacier.ky
    kz = glacier.kz
    IBOUND = glacier.IBOUND
    FQ = glacier.FQ
    HI = glacier.HI
    upper_surface = glacier.upper_surface
    lower_surface = glacier.lower_surface
    WaterTable = glacier.WaterTable
    porosity = glacier.porosity
    Ss = glacier.storage
    cryoconite_locations = glacier.cryoconite_locations


    # determine shape of array from lengths of each dimension
    Nz, Ny, Nx = len(z)-1, len(y)-1, len(x)-1
    Nod = np.prod(SHP) # Nod is the total number of cells in the grid (x.y.z)


    # set up output arrays
    # number of timesteps is one less than length t due to zero-indexing
    Nt = len(t)-1    
    Out.Phi = np.zeros((Nt+1, Nod)) # Nt+1 times
    Out.Q   = np.zeros((Nt  , Nod)) # Nt time steps
    Out.Qs  = np.zeros((Nt  , Nod))
    Out.Qx  = np.zeros((Nt, Nz, Ny, Nx-1))
    Out.Qy  = np.zeros((Nt, Nz, Ny-1, Nx))
    Out.Qz  = np.zeros((Nt, Nz-1, Ny, Nx))
    Out.sf = np.zeros((Nt,Nz-1,Ny,Nx))
    Out.BBA = np.zeros((Nt,Ny,Nx))
    Out.melt = np.zeros((Nt,Nz,Ny,Nx))
    # array where each element is the step size between array values,
    # reshaped into vectors
    # e.g. for array [0,2,4,6], diff = [2,2,2,2]

    dx = np.abs(np.diff(x).reshape(1, 1, Nx))
    dy = np.abs(np.diff(y).reshape(1, Ny, 1))
    dz = np.abs((np.diff(z)).reshape(Nz, 1, 1))

    # create boolean arrays for active, inactive and fixed-head cells
    active = (IBOUND>0).reshape(Nod) #vector of booleans
    inact  = (IBOUND==0).reshape(Nod) # dito for inact
    fxhd   = (IBOUND<0).reshape(Nod) # dito for fxhd


    for idt, dt in enumerate(np.diff(t)):

        # the iterator is a tuple - the first element (idt) is the index or "step number" (0,1,2,3)
        # while the second element (dt) is equal to the step size, therefore, 
        # idt is the current timestep and it is set to the following timestep (idt+1)
        # timestep
        it = idt + 1

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
        # in each dimension there are faces that do not have neighbours at either edge
        # this will be replicated in the cell number arrays IE,IW,IS...

        Cx = 1 / (Rx1[  :, :,  :-1] + Rx2[:, : , 1:])
        Cy = 1 / (Ry1[  :, :-1,:  ] + Ry2[:, 1:, : ])
        Cz = 1 / (Rz1[:-1, :,  :  ] + Rz2[1:, :, : ])

        # storage term, variable dt not included
        Cs = (Ss*(dx*dy*dz) / epsilon).ravel()

        # reshape Nod to SHP
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
        # first ravelled and then concatenated into one continuous 1D vector. Then the arrays of
        # N, S, E, W, U and D neighbour indexes (IE, IW...) are treated identically to
        # produce vectors of neighbour indices that aligns with the corresponding
        # conductance value C. For eample, IW, IE gives the indexes of the L and R neighbours
        # of cell i for which the conductance is given in Cx. This is all organised into a scipy
        # sparse matrix which stores the row and column indices in a tuple and the associated
        # conductance value in "data". The final tuple in the sparse matrix constructor reshapes the
        # sparse matrix to (Nod,Nod) which gives row length and column length both to be equal to 
        # the number of possible cell combinations in the finite difference grid. Therefore, for 
        # every possible combination of row and column index there is a an associated value for the 
        # conductance in each direction. For most combinations of rows and columns, the C values 
        # in all directions will be zero because they are not within the local (6 cell) neighbourhood 
        # of any particular cell. The tuple of row and column indices passed to the sparse matrix
        # constructor defines those cells that have non-zero conductance (i.e. combinations of cells
        # that are within a 6-cell neighbourhood). A list of all individual cells in the finite 
        # difference grid s found by following the matrix primary diagonal.

        A = sp.csc_matrix(( np.concatenate(( R(Cx), R(Cx), R(Cy), R(Cy), R(Cz), R(Cz))),
        (np.concatenate(( R(IE), R(IW), R(IN), R(IS), R(IB), R(IT)) ),\
            np.concatenate(( R(IW), R(IE), R(IS), R(IN), R(IT), R(IB)) ))),(Nod,Nod))

        # Shape of A is ((x * y * z),( x * y * z)) because for every cell A stores
        # its conductance relation with every other cell in the finite difference grid. IE, IW... are
        # arrays shifted by 1 in each dimensions and therefore represent the pairs of indices for 
        # adjacent cells (i.e. non-zeros conductance). The total conductance out of a cell is the 
        # row-wise sum of conductances Cx (L) + Cx (R) + Cy (UP) + Cy (DP) + Cz (UP) + Cz (DN). 
        # 
        # Since the square matrix has a row for each cell and a column for each cell, the off-diagonal
        # elements represent the conductance relations between each cell and all others. This means the 
        # diagonal elements remain empty because they represent the reltion of the cell with itself.
        # We then fill the empty diagonal with the row sums, because this is equivalent to assigning
        # the total conductance to the cell.
        # 
        # The physical meaning of the diagonal matrix element is the amount of water flowing from 
        # node i to all its adjacent nodes if the head in node i is exactly 1 m higher than that 
        # of its neighbors. 

        A = -A + sp.diags(np.array(A.sum(axis=1))[:,0])
        
        # reshape input arrays to vectors using our ravel shorthand R
        # to enable vector multiplication with system matrix A

        FQ = R(FQ);  HI = R(HI);  Cs = R(Cs)

        # initialize heads
        Out.Phi[0] = HI

        # solve heads at active locations at t_i+eps*dt_i

        Nt=len(t)  # for heads, at all times Phi at t[0] = initial head

        Ndt=len(np.diff(t)) # for flows, average within time step

        # compute right hand side of equation as the dot product of matrix A and vector Phi (head at previous timestep)
        # FQ is a vector of extraction terms, Cs is a vector of storage terms, slicing to [:,fxhd] restricts the 
        # computation to the fixed heads in the grid - this effectively adjusts RHS to take account of flows in/out of
        # fixed head nodes as well as storage and source/extraction terms that are buried in FQ. The result is a complete
        # RHS term that can be used to solve the matrix computation for the unknown head values in the next time step.

        # i.e. RHS = source_and_extraction - (system_matrix + storage_on_diagonal) dot (hydraulic head at previous timestep)
        RHS = FQ - (A + sp.diags(Cs / dt))[:,fxhd].dot(Out.Phi[it-1][fxhd]) 

        # use RHS (computed from knowns in previous timestep) to solve matrix computation for unknown heads (vector Phi) in next time step
        # The matrix is restricted to the active cells.

        # i.e. head_now = solve(matrixA + storage_on_diagonal, RHS + storage / head_then * timestep)
        Out.Phi[it][active] = spsolve( (A + sp.diags(Cs / dt))[active][:,active], RHS[active] + Cs[active] / dt*Out.Phi[it-1][active])

        # calculate net flow into cell as dot product of system matrix and current head values
        Out.Q[idt]  = A.dot(Out.Phi[it])

        # calculate net flow released from storage as change in head multiplied by specific storage
        Out.Qs[idt] = -Cs/dt*(Out.Phi[it]-Out.Phi[it-1])

        # calculate flows across cell faces in each dimension by taking flows in individual dimensions
        Out.Qx[idt] =  -np.diff( Out.Phi[it].reshape(SHP), axis=2)*Cx
        Out.Qy[idt] =  +np.diff( Out.Phi[it].reshape(SHP), axis=1)*Cy
        Out.Qz[idt] =  +np.diff( Out.Phi[it].reshape(SHP), axis=0)*Cz

        # update head to end of time step
        Out.Phi[it] = Out.Phi[it-1] + (Out.Phi[it] - Out.Phi[it-1]) / epsilon

        # impose limit on hydraulic head - if head descends below bottom boundary, set to bottom boundary
        # if head rises above upper surface, set to upper surface. Any excess head considered as surface runoff

        upper = upper_surface.ravel()
        lower = lower_surface.ravel()

        if constrain_head_to_WC:
            
            for i in range(Nz+1):
                
                if i==0:
                    Phi = Out.Phi[it].ravel()
                    Phi[0:len(upper)] = np.where(Phi[0:len(upper)]>upper,upper,Phi[0:len(upper)])
                    Phi[0:len(lower)] = np.where(Phi[0:len(lower)]<lower,lower,Phi[0:len(lower)])
                    Out.Phi[it] = Phi

                elif (i > 0) & (i < Nz):
                    Phi = Out.Phi[it].ravel()
                    Phi[len(upper)*i:len(upper)*(i+1)] = np.where(Phi[len(upper)*i:len(upper)*(i+1)]>upper,upper,Phi[len(upper)*i:len(upper)*(i+1)])
                    Phi[len(lower)*i:len(lower)*(i+1)] = np.where(Phi[len(lower)*i:len(lower)*(i+1)]<lower,lower,Phi[len(lower)*i:len(lower)*(i+1)])
                    Out.Phi[it] = Phi

                elif i == Nz:
                    Phi = Out.Phi[it].ravel()
                    Phi[len(Phi)-len(upper):len(Phi)] = np.where(Phi[len(Phi)-len(upper):len(Phi)]>upper,upper,Phi[len(Phi)-len(upper):len(Phi)])
                    Phi[len(Phi)-len(lower):len(Phi)] = np.where(Phi[len(Phi)-len(lower):len(Phi)]<lower,lower,Phi[len(Phi)-len(lower):len(Phi)])
                    Out.Phi[it] = Phi

                else: 
                    print("Nz out of range")


        if MELT_CALCS:

            #########################
            # BEGIN MELT CALCULATIONS

            # First calculate the effective grain diameters
            density = (1-glacier.porosity) * 917 # bulk density is 1-porosity * pure-ice density
            SSA = 17.6 - (0.02 * density)
            reff = (6/(SSA / 917))/2 
            
            # round reff and density down to nearest 100 for consistency with snicar LUT
            reff = (reff / 100).astype(int) * 100    
            reff = np.where(reff>=1500,1500,reff)
            density = (density/100).astype(int)*100
            
            # load appropriate LUT depending on user-defined algal load
            # algae 1 - 5 = 5000, 10000, 25000, 50000, 100000 ppb
            # algae 0 = clean ice

            if glacier.algae == 0:
                LUT = np.load('SNICAR_LUT.npy')
            elif glacier.algae == 1:
                LUT = np.load('SNICAR_LUT_alg1.npy')
            elif glacier.algae == 2:
                LUT = np.load('SNICAR_LUT_alg2.npy')
            elif glacier.algae == 3:
                LUT = np.load('SNICAR_LUT_alg3.npy')
            elif glacier.algae == 4:
                LUT = np.load('SNICAR_LUT_alg4.npy')
            elif glacier.algae == 5:
                LUT = np.load('SNICAR_LUT_alg5.npy')
            
            else:
                raise("Value for algal loading is invalid: please choose integer between 0 and 3")

            # ravel the input vars to help with common indexing
            albedo = np.zeros((SHP[1],SHP[2]))
            reff = reff.ravel()
            density = density.ravel()

            # loop through pixels and grab the albedo from the var idxs
            # the variable indexes are the nearest integer to their
            # values /100 -1.

            for a in range(len(albedo)):
                idx_rds = int(reff[a]/100)-1
                idx_dens = int(density[a]/100)-1
                albedo[a] = LUT[idx_rds,idx_dens]

            # add BBA to the Out n-tuple                    
            Out.BBA[it-1] = albedo.reshape(SHP[1],SHP[2])

                       
            @dask.delayed
            def runit(alb, glacier):
                
                lon_ref = 0
                summertime = 0
                albedo = alb
                roughness = 0.005
                met_elevation = glacier.elevation

                SWR,LWR,SHF,LHF = ebm.calculate_seb(glacier.lat, glacier.lon, lon_ref,\
                     glacier.day, glacier.time, summertime, glacier.slope, glacier.aspect,\
                          glacier.elevation, met_elevation, glacier.lapse, glacier.inswrd,\
                              glacier.avp, glacier.airtemp, glacier.windspd, albedo,\
                                 glacier.roughness)

                sw_melt, lw_melt, shf_melt, lhf_melt, total = ebm.calculate_melt(
                    SWR,LWR,SHF,LHF, glacier.windspd, glacier.airtemp)

                return sw_melt, shf_melt

            rmelt,tmelt = runit(albedo, glacier).compute()
            tmelt = np.zeros(shape=(rmelt.shape))+tmelt

              # reshape the ravelled melt values to grid shape
            rmelt = np.reshape(np.array(rmelt),[Ny,Nx])
            tmelt = np.reshape(np.array(tmelt),[Ny,Nx])
            
            ### NOW DISTRIBUTE RAD MELTING OVER VERTICAL LAYERS
            # melt is only predicted for surface
            # here, calculate the melt at each cell center
            # this is done by approximating an exponential decay
            # by dividing the surface melt by the factorial of the 
            # depth (i.e. at depth = 3, melt = surface_melt / (3+2+1)
            # at the cell center, the melt is the mean of the upper
            # and lower boundary melt.

            # melt3d will be melt at each vertical cell boundary
            melt3d = np.zeros(shape =(len(z),Ny,Nx))
            # melt_at_cell_centers is melt midway between lower and upper cell boundary
            melt_at_cell_centers = np.zeros(shape =(Nz,Ny,Nx))
            # upper surface is melt predicted by ebmodel
            melt3d[0,:,:] = rmelt

            # for each vertical layer boundary melt is divided by factorial
            # of depth below surface
            for p in np.arange(1,len(z),1):
                melt3d[p,:,:] = rmelt/factorial(p)

            # melt at cell centers is mean of melt at upper and lower boundaries
            for q in np.arange(0,Nz,1): 
                melt_at_cell_centers[q,:,:] = (melt3d[q,:,:]+melt3d[q+1,:,:])/2
            
            # send cell center melt to Out n-tuple
            Out.melt[it-1] = melt_at_cell_centers

            porosity += (melt_at_cell_centers/1000) # radiative flux increases porosity
            porosity[0,:,:] -= (tmelt/1000)  # turbulent flux decreases porosity (at surface)
            
            FQ += melt_at_cell_centers.ravel() # + rmelt vol to hydro source term for next timestep
            FQ[0:len(tmelt.ravel())] +=tmelt.ravel() # add surface lowering vol to source term 
            ### WORK OUT VOLUME CHANGE AND UPDATE POROSITY
            ### THEN CREATE MORE LUTs FOR ALGAL CONCNs
            
            ### make sure rmelt + tmelt feeds back into F term


    # reshape Phi to shape of grid
    Out.Phi = Out.Phi.reshape((Nt,) + SHP)
    Out.Q   = Out.Q.reshape( (Ndt,) + SHP)
    Out.Qs  = Out.Qs.reshape((Ndt,) + SHP)

    if moulin_location != None:
        Out.Phi[it,:,moulin_location[0][0]:moulin_location[0][1],moulin_location[1][0]:moulin_location[1][1]] = 0


    return Out, porosity # all outputs in a named tuple for easy access
