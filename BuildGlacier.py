import numpy as np

class Glacier:

    def __init__(self, x, y, z, cell_spacing_xy, cell_spacing_z, base_elevation, WC_thickness0, porosity0, specific_retention, WaterTable0,\
    cryoconite_coverage, melt_rate0, rainfall0, slope, kxy, kz, loss_at_edges, loss_at_terminus,\
    stream_location, moulin_location, moulin_extr_rate, algae, lat, lon, day, time, aspect,\
        roughness, lapse, windspd, airtemp, inswrd, avp):

        SHP = (len(z)-1, len(y)-1, len(x)-1)

        # derived variables
        upper_surface = (np.zeros(SHP[1:]) + base_elevation)  + np.random.rand(SHP[1],SHP[2])/10
        lower_surface = (upper_surface - WC_thickness0) + np.random.rand(SHP[1],SHP[2])/10
        WaterTable = lower_surface + (WC_thickness0*WaterTable0)
        cryoconite_locations = np.random.choice(a=[True,False],size=SHP[1:], p=[cryoconite_coverage,1-cryoconite_coverage])
        melt_rate = np.zeros(SHP)+melt_rate0
        rainfall = np.zeros(SHP)+rainfall0
        porosity = np.ones(SHP)*porosity0
        Ss = porosity - specific_retention 
        elevation = base_elevation

        for i in range(upper_surface.shape[0]):

            upper_surface[i,:] += (upper_surface.shape[0] - (i*slope))/10
            lower_surface[i,:] += (lower_surface.shape[0] - (i*slope))/10
            WaterTable[i,:] += (WaterTable.shape[0] - (i*slope))/10
        
        #3D water table
        HI = np.zeros(SHP)
        HI[0,:,:] = WaterTable0

        # calculate hydraulic head at each finite difference length beneath 
        # the water table surface
        for i in np.arange(1,WC_thickness0/cell_spacing_z-1,1):
            HI[int(i),:,:] = WaterTable0 - cell_spacing_z + (cell_spacing_z*i)

        # set hydraulic conductivity in m/d - these are 3D arrays where the conductivity through the horizontal (x), 
        # horizontal (y) or vertical (z) faces are defined - glacier average from Stevens et al (2018) = 0.185
        kx = (np.random.rand(len(z)-1,len(y)-1,len(x)-1)/10)+kxy # [m/d] 3D kx array
        ky = (np.random.rand(len(z)-1,len(y)-1,len(x)-1)/10)+kxy # [m/d] 3D ky array with same values as kx
        kz = (np.random.rand(len(z)-1,len(y)-1,len(x)-1)/10)+kz # [m/d] 3D kz array with same values as kx

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
        FQ += rainfall

        IBOUND = np.ones(SHP)
        IBOUND[:, -1, :] = 0 # last row of model heads are prescribed (-1 head at base boundary)
        IBOUND[:, 0, :] = 0 # these cells are inactive (top boundary)

        self.SHP = SHP
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.IBOUND = IBOUND
        self.FQ = FQ
        self.HI = HI
        self.upper_surface = upper_surface
        self.lower_surface = lower_surface
        self.WaterTable = WaterTable
        self.porosity = porosity
        self.storage = Ss
        self.cryoconite_locations = cryoconite_locations
        self.algae = algae
        self.lat = lat
        self.lon = lon
        self.day = day
        self.time = time
        self.slope = slope
        self.roughness = roughness
        self.lapse = lapse
        self.windspd = windspd
        self.airtemp = airtemp
        self.inswrd = inswrd
        self.avp = avp
        self.elevation = elevation
        self.aspect = aspect

        return 