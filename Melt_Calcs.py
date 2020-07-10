import dask
import numpy as np
import ebmodel as ebm
from math import factorial


def melt_calcs(glacier, Out, Nx, Ny, Nz, SHP, it, idt, x, y, z, porosity, datapath):

    #########################
    # BEGIN MELT CALCULATIONS

    algal_map = glacier.algal_map[0,:,:].ravel()

    # First calculate the effective grain diameters
    density = (1-porosity) * 917 # bulk density is 1-porosity * pure-ice density
    SSA = 17.6 - (0.02 * density)
    reff = (6/(SSA / 917))/2 
    
    # round reff and density down to nearest 100 for consistency with snicar LUT
    reff = (reff / 100).astype(int) * 100    
    reff = np.where(reff>=1500,1500,reff)
    density = (density/100).astype(int)*100
    
    # load appropriate LUT depending on user-defined algal load
    # algae 1 - 5 = 5000, 10000, 25000, 50000, 100000 ppb
    # algae 0 = clean ice
    
    LUT0 = np.load(str(datapath + 'SNICAR_LUT_alg0.npy'))
    LUT1 = np.load(str(datapath + 'SNICAR_LUT_alg1.npy'))
    LUT2 = np.load(str(datapath + 'SNICAR_LUT_alg2.npy'))
    LUT3 = np.load(str(datapath + 'SNICAR_LUT_alg3.npy'))
    LUT4 = np.load(str(datapath + 'SNICAR_LUT_alg4.npy'))
    LUT5 = np.load(str(datapath + 'SNICAR_LUT_alg5.npy'))

    # ravel the input vars to help with common indexing
    albedo = np.zeros((SHP[1],SHP[2])).ravel()
    reff = reff.ravel()
    density = density.ravel()

    # loop through pixels and grab the albedo from the var idxs
    # the variable indexes are the nearest integer to their
    # values /100 -1.

    for a in range(len(albedo)):
        
        idx_rds = int(reff[a]/100)-1
        idx_dens = int(density[a]/100)-1

        if algal_map[a] == 0:
            albedo[a] = LUT0[idx_rds,idx_dens]
        elif algal_map[a] == 1:
            albedo[a] = LUT1[idx_rds,idx_dens]
        elif algal_map[a] == 2:
            albedo[a] = LUT2[idx_rds,idx_dens]
        elif algal_map[a] == 3:
            albedo[a] = LUT3[idx_rds,idx_dens]
        elif algal_map[a] == 4:
            albedo[a] = LUT4[idx_rds,idx_dens]
        elif algal_map[a] == 5:
            albedo[a] = LUT5[idx_rds,idx_dens]

    albedo = albedo.reshape((SHP[1],SHP[2]))
    
    # set albedo of cryoconite holes to 0.4 (accounting for overlying water etc)    
    albedo = np.where(glacier.cryoconite_locations,0.4,albedo)
    
    # add BBA to the Out n-tuple                    
    Out.BBA[it-1] = albedo.reshape(SHP[1],SHP[2])
    
    @dask.delayed
    def runit(alb, glacier):
        
        lon_ref = 0
        summertime = 0
        albedo = alb
        roughness = glacier.roughness
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


    return Out, rmelt, tmelt, melt3d, melt_at_cell_centers, porosity