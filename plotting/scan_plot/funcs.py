def cart2pol(x, y):
    import numpy as np
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (phi, rho)

def pol2cart(phi, rho):
    import numpy as np
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def convert_to_datetime(row):
    import datetime as dt
    currDateStr = str( int( row["dateStr"] ) )
#     return currDateStr
    if row["timeStr"] < 10:
        currTimeStr = "000" + str( int( row["timeStr"] ) )
    elif row["timeStr"] < 100:
        currTimeStr = "00" + str( int( row["timeStr"] ) )
    elif row["timeStr"] < 1000:
        currTimeStr = "0" + str( int( row["timeStr"] ) )
    else:
        currTimeStr = str( int( row["timeStr"] ) )
    return dt.datetime.strptime( currDateStr\
                    + ":" + currTimeStr, "%Y%m%d:%H%M" )

def ace_read(sTime, eTime, res=1):
    import numpy as np
    from spacepy import pycdf
    import datetime as dt
    import bisect
    import pandas as pd
    from glob import glob

    Bdata = np.zeros([1,3])
    Btime = np.zeros(1)
    Vsw_data = np.zeros([1,3])
    Vsw_time = np.zeros(1)
    Np_data = np.zeros(1)
    Np_time = np.zeros(1)
    if (eTime-sTime).days > 0:
        ndays = (eTime-sTime).days
    else:
        ndays = 1
    for i in range(ndays): 
        ace_date = (sTime+dt.timedelta(days=i)).strftime("%Y%m%d")
        ace_year = (sTime+dt.timedelta(days=i)).strftime("%Y")
        ace_mfi_file = glob('/sd-data/ace/' + ace_year + '/ac_h0_mfi_' + ace_date + '*.cdf')[0]
        ace_swe_file = glob('/sd-data/ace/' + ace_year + '/ac_h0_swe_' + ace_date + '*.cdf')[0]
        ace_mfi = pycdf.CDF(ace_mfi_file)
        ace_swe = pycdf.CDF(ace_swe_file)

        # set start and stop positions 
        start = sTime
        stop = eTime 
        start_ind_mfi = bisect.bisect_left(ace_mfi['Epoch'], start)
        stop_ind_mfi = bisect.bisect_left(ace_mfi['Epoch'], stop)
        start_ind_swe = bisect.bisect_left(ace_swe['Epoch'], start)
        stop_ind_swe = bisect.bisect_left(ace_swe['Epoch'], stop)
        # grab the data we want
        data = ace_mfi['BGSM'][start_ind_mfi:stop_ind_mfi]
        time1 = ace_mfi['Epoch'][start_ind_mfi:stop_ind_mfi]
        Vsw = ace_swe['V_GSM'][start_ind_swe:stop_ind_swe]
        time2 = ace_swe['Epoch'][start_ind_swe:stop_ind_swe]
        Np = ace_swe['Np'][start_ind_swe:stop_ind_swe]
        time3 = time2
        Bdata = np.append(Bdata,data, axis=0)
        Btime = np.append(Btime,time1)
        Vsw_data = np.append(Vsw_data, Vsw, axis=0)
        Vsw_time = np.append(Vsw_time, time2)
        Np_data = np.append(Np_data, Np)
        Np_time = np.append(Np_time, time3)

        ace_mfi.close()
        ace_swe.close()

    # remove the zeros in the first row of numpy arrays
    Bdata = Bdata[1:,:]
    Btime = Btime[1:]
    Vsw_data = Vsw_data[1:,:]
    Vsw_time = Vsw_time[1:]
    Np_data = Np_data[1:]
    Np_time = Np_time[1:]

    time_mfi = Btime
    time_swe = Vsw_time

#    # remove non-phisical values
#    B_total = (Bdata[:, 0]**2 + Bdata[:, 1]**2 + Bdata[:, 2]**2)**0.5
#    B_total_ind = np.where(np.abs(B_total) < 1e2)
#    B_total = B_total[B_total_ind]
#    Btime_total = Btime[B_total_ind]
#    Bx_data = Bdata[:,0]
#    Bx_ind = np.where(np.abs(Bx_data) < 1e2)
#    Bx_data = Bx_data[Bx_ind]
#    Bx_time = Btime[Bx_ind]
#    By_data = Bdata[:,1]
#    By_ind = np.where(np.abs(By_data) < 1e2)
#    By_data = By_data[By_ind]
#    By_time = Btime[By_ind]
#    Bz_data = Bdata[:,2]
#    Bz_ind = np.where(np.abs(Bz_data) < 1e2)
#    Bz_data = Bz_data[Bz_ind]
#    Bz_time = Btime[Bz_ind]
#
#    Vsw_ind = np.where(np.abs(Vsw_data[:,0]) < 1e4)
#    Vsw_data = Vsw_data[Vsw_ind]
#    Vsw_time = Vsw_time[Vsw_ind]

    df_mfi = pd.DataFrame(data=Bdata, index=Btime, columns=['Bx', 'By', 'Bz'])
    df_swe = pd.DataFrame(data=np.append(Vsw_data, Np_data.reshape(Np_data.shape[0], 1), axis=1),
                          index=Vsw_time, columns=['Vx', 'Vy', 'Vz', 'Np'])

    if res==1: 
        df_swe.set_index([[dd.replace(second=0) for dd in df_swe.index]], inplace=True)
        df_mfi.set_index([[dd.replace(second=0) for dd in df_mfi.index]], inplace=True)
        # take the one minute average 
        df_mfi = df_mfi.groupby(df_mfi.index).median()
        # joint dfs to a single df
        df_ace = df_mfi.join(df_swe, how='outer')

        return df_ace

    else:
        return df_mfi, df_swe

def return_nan_if_IndexError(data, index):
    """ returns np.nan if index is out of range of data,
    else returns data[index].

    Parameters
    ----------
    data : list or np.array
    index : int

    Returns
    data[index] or np.nan
    
    """
    import numpy as np
    try:
	out = data[index]
    except IndexError:
	out = np.nan

    return out

def create_gridpnts(lat_min=60,lat_max=90,dlat=1, half_dlat_offset=False):
    '''
    "half_dlat_offset=False" implements NINT[360 sin(theta)] at theta = 89, 88, ... colatitude
    "half_dlat_offset=True" implements NINT[360 sin(theta)] at theta = 89.5, 88.5, ... colatitude
    '''

    import numpy as np
    lats_cntr = [x + 0.5*dlat for x in range(lat_min,lat_max,dlat)] 
    if half_dlat_offset:
        nlons = [round(360 * np.sin(np.deg2rad(90-lat))) for lat in lats_cntr]
    else:
        nlons = [round(360 * np.sin(np.deg2rad(90-(lat-0.5*dlat)))) for lat in lats_cntr]
    dlons = [360./nn for nn in nlons]

    # lat and lon bins
    lat_bins = [x for x in np.arange(lat_min,lat_max+dlat,dlat)] 
    lon_bins = []
    lons_cntr = []      # list of lists of lons
    for i in range(len(lats_cntr)):
        lon_tmp = [ item*dlons[i] for item in np.arange(0.5, nlons[i]+0.5) ]
        lons_cntr.append(lon_tmp)
        lon_tmp = [ item*dlons[i] for item in np.arange(nlons[i]) ]
        lon_tmp.append(360) 
        lon_bins.append(lon_tmp)

    return lons_cntr, lats_cntr, lon_bins, lat_bins 

def grid_sddata(rads,rads_data=None, fovs=None, sites=None, min_range_lim= 0, npnts_minlim=3,
        stime=None,interval=None,fileType=None,filtered=None,channel=None,coords=None, 
        lat_min=60,lat_max=90,dlat=1,half_dlat_offset=False):

    import matplotlib.pyplot as plt
    import datetime as dt
    import pandas as pd
    import numpy as np
    from davitpy import pydarn
    from davitpy.utils.coordUtils import coord_conv

    if (rads_data is None) or (fovs is None) or (sites is None):
        from davitpy.pydarn.sdio.radDataRead import radDataOpen, radDataReadRec
        #open the data files
        myFiles = []
        for i in range(len(rads)):
            f = radDataOpen(stime,rads[i],stime+dt.timedelta(seconds=interval),fileType=fileType,filtered=filtered,channel=channel)
            if(f is not None): 
                myFiles.append(f)

        allBeams = [''] * len(myFiles)
        sites,fovs=[],[]
        #go through all open files
        for i in range(len(myFiles)):
            #read until we reach start time
            allBeams[i] = radDataReadRec(myFiles[i])
            #while (allBeams[i].time < stime and allBeams[i] is not None):
            while ((allBeams[i] is not None) and (allBeams[i].time < stime)):

                allBeams[i] = radDataReadRec(myFiles[i])

            #check that the file has data in the target interval
            if(allBeams[i] is None): 
                myFiles[i].close()
                myFiles[i] = None
                fovs.append(None)
                sites.append(None)
                continue

            t=allBeams[i].time
            site = pydarn.radar.site(radId=allBeams[i].stid,dt=t)
            sites.append(site)
            # create fov object
            myFov = pydarn.radar.radFov.fov(site=site, rsep=allBeams[i].prm.rsep,\
                                            ngates=allBeams[i].prm.nrang+1,
                                            nbeams=site.maxbeam, coords=coords,
                                            date_time=t)
            fovs.append(myFov)
        etime = stime + dt.timedelta(seconds=interval)
        #go though all files
        rads_data = [[] for i in range(len(myFiles))]
        for i in range(len(myFiles)):
            #check that we have good data at this time
            if(myFiles[i] is None):
                rads_data[i] = None
                continue
            #until we reach the end of the time window
            while(allBeams[i] is not None and allBeams[i].time < etime):
                rads_data[i].append(allBeams[i])
                #read the next record
                allBeams[i] = radDataReadRec(myFiles[i])
            if rads_data[i] == []:
                rads_data[i] = None

    # create grid points
    lons_cntr, lats_cntr, lon_bins, lat_bins = create_gridpnts(lat_min=lat_min,
            lat_max=lat_max,dlat=dlat,half_dlat_offset=half_dlat_offset)
    # grid the data
    df_lst = []
    rads_lft = []
    for i in range(len(rads)):
        rad_data = rads_data[i]
        if rad_data is None:
            continue
        fov = fovs[i]
        # gridded parameters
        gvels, glats, glons, gbmazms = [], [], [], [] 
        for myBeam in rad_data:
            if (len(myBeam.fit.slist)==0) or (myBeam is None): continue
            for k in range(0,len(myBeam.fit.slist)):
                if myBeam.fit.slist[k] not in fov.gates:
                    continue
                if (myBeam.fit.slist[k] * myBeam.prm.rsep) < min_range_lim:
                    continue
#                if (myBeam.fit.gflg[k]):     # filter out ground scatter
#                    continue
                r = myBeam.fit.slist[k]
                ilon = fov.lonCenter[myBeam.bmnum,r]
                ilat = fov.latCenter[myBeam.bmnum,r]
                ivel = myBeam.fit.v[k]
                bmazm = myBeam.prm.bmazm

		# grid lat
                indx_lat = np.digitize([ilat], lat_bins)
		indx_lat = indx_lat[0]-1
                #glat = lats_cntr[indx_lat]

                # NOTE: the following way of using return_nan_if_IndexError
                # avoids nan in latc
                glat = return_nan_if_IndexError(lats_cntr, indx_lat)
                glats.append(glat)

		# grid lon
		# NOTE: the following way avoids nan in lonc
		try: 
		    indx_lon = np.digitize([ilon % 360], lon_bins[indx_lat])
		    indx_lon = indx_lon[0]-1
		    glon = lons_cntr[indx_lat][indx_lon]
		except IndexError:
                    glon = np.nan
                glons.append(glon)

                gvels.append(ivel)
                gbmazms.append(bmazm)

        rads_lft.append(rads[i])
        rad_geolat = sites[i].geolat
        rad_geolon = sites[i].geolon
        rad_lat = [rad_geolat]*len(glats)
        rad_lon = [rad_geolon]*len(glons)

        #columns = ['latc', 'lonc', 'lat_'+rads[i], 'lon_'+rads[i], 'vel_'+rads[i], 'bmazm_'+rads[i]]
        columns = ['latc', 'lonc', 'geolat_rad', 'geolon_rad', 'vel', 'bmazm']
        df_tmp = pd.DataFrame(zip(glats, glons, rad_lat, rad_lon, gvels, gbmazms), columns=columns)
        if (not df_tmp.empty):
            df_tmp = df_tmp.sort(['latc', 'lonc'])
            df_tmp = df_tmp.groupby(['latc', 'lonc'], as_index=False).filter(lambda x: len(x)>= npnts_minlim)
            df_tmp = df_tmp.groupby(['latc', 'lonc'], as_index=False).median()
        if (not df_tmp.empty):
            df_lst.append(df_tmp)
    if df_lst==[]:
        df = None
    else:
        df = pd.concat(df_lst, keys=rads_lft)
    return df

def calc_azm(myMap, df):
    """
    Calculates the azimuth angle of an LOS vector with
    respect to the pole in myMap.coords system
    """
    import numpy as np

    coords = myMap.coords
    time = myMap.datetime
    dfg = df.groupby(level=0, sort=False)

    azms = []
    # North pole in the myMap.coords system 
    xp, yp = myMap(0, 90)
    for index, group in dfg:
        geolon_rad, geolat_rad = group.geolon_rad[0], group.geolat_rad[0]
        xr, yr = myMap(geolon_rad, geolat_rad, coords='geo')
        for indx, row in group.iterrows():
            xc, yc = myMap(row['lonc'],row['latc'])
            the_bm = np.arctan2(yc-yr, xc-xr)
            the_p = np.arctan2(yp-yc, xp-xc)
            azm_los = np.rad2deg(the_p - the_bm)
            azms.append(azm_los)

    #df['azm_los'] = np.array(azms)
    df.loc[:, 'azm_los'] = np.array(azms)
    return df

def sdvel_lfit(myMap, df, npntslim_lfit=5, OLS=False):
    """
    Resolves the 2D flow vector using L-shell cosine fitting method
    """
    import pandas as pd
    import scipy

    # groupby by the center latitude of the grids and filter out the
    # latitudinal grids that has less points than the npntslim_lfit argument
    df = df.groupby(['latc'], as_index=False).\
            filter(lambda x: len(x) >= npntslim_lfit)
    if (df is not None) and (not df.empty):
        # calculate the LOS azm angle at the grid point,
        # which is bewtween LOS direction and pole direction
        df = calc_azm(myMap, df)

        # df that includes lfit parameters
        if OLS:
            df = df.groupby(['latc'], as_index=False).apply(lfit_OLS)
        else:
            df = df.groupby(['latc'], as_index=False).apply(lfit_non_linear_LS)
    else:
        df = None
    return df

def lfit_OLS(group):
    """Performs OLS fit
    This function is called inside sdvel_lfit
    """

    import statsmodels.api as sm 
    import numpy as np

    A = np.column_stack((np.cos(np.radians(group.azm_los.as_matrix())),\
                         np.sin(np.radians(group.azm_los.as_matrix()))))

    yy = group.vel.as_matrix()
    model = sm.OLS(yy, A)
    reslt3 = model.fit()
    velN, velE = reslt3.params
    velN_err, velE_err = reslt3.bse
   ## double checking using other lsqr fitting modules
   #reslt1 = np.linalg.lstsq(A, yy)
   #velN, velE = reslt1[0]
   #reslt2 = scipy.sparse.linalg.lsqr(A,yy)
   #velN, velE = reslt2[0]
   ##A = sm.add_constant(A)
    group['velN'], group['velN_err'] = velN, velN_err
    group['velE'], group['velE_err'] = velE, velE_err
    group['lfit_vel'] = np.sqrt(group.velN.apply(np.square) +\
                                group.velE.apply(np.square))
    group['lfit_vel_err'] = np.sqrt(group.velN_err.apply(np.square) +\
                                group.velE_err.apply(np.square))
    group['lfit_azm'] = 90. - np.degrees(np.arctan2(velN, velE))

    return group

def lfit_non_linear_LS(group):
    """Performs non-linear LS fit
    This function is called inside sdvel_lfit
    """

    import numpy as np

    # do cosine fitting with weight
    azm = group.azm_los.as_matrix()
    los_vel = group.vel.as_matrix()
    sigma = sigma =  np.array([1.0 for x in azm])
    fitpars, perrs = cos_curve_fit(azm, los_vel, sigma)
    vel_mag = round(fitpars[0],2)
    vel_dir = round(np.rad2deg(fitpars[1]) % 360,1)
    vel_mag_err = round(perrs[0],2)
    vel_dir_err = round(np.rad2deg(perrs[1]) % 360, 1)

    group['lfit_vel'] = vel_mag 
    group['lfit_vel_err'] = vel_mag_err
    group['lfit_azm'] = vel_dir

    return group

def cosfunc(x, Amp, phi):
    import numpy as np
    return Amp * np.cos(1 * x - phi)

def cos_curve_fit(azms, vels, sigma):
    import numpy as np
    from scipy.optimize import curve_fit
    fitpars, covmat = curve_fit(cosfunc, np.deg2rad(azms), vels, sigma=sigma)
    perrs = np.sqrt(np.diag(covmat)) 

    return fitpars, perrs



################# sdvel_cosfit() has to be modified ################
def sdvel_cosfit(df, npntslim_cosfit=5):
    import pandas as pd
    import numpy as np
    # find the bmazms relative to a radar that is located in the westmost direction, and then delete geolon_rad and geolat_rad columns
    geolon_rad_min = df['geolon_rad'].min()
    df['bmazm'] = df['bmazm'] - (df['geolon_rad'] - geolon_rad_min) 
    df = df.drop(['geolat_rad', 'geolon_rad'], axis=1)
    # remove the top level of the hierarchical indexes, which is the keys for radar codes
    rads = [rad for rad in df.index.levels[0]]
    df = pd.concat([df.ix[rad] for rad in df.index.levels[0]], ignore_index=True)

    # groupby by the center latitude of the grids and filter out the latitudinal grids that has less points than the npntslim_cosfit argument
    #df = df.groupby(['latc']).sort['lonc']
    df = df.groupby(['latc'], as_index=False).filter(lambda x: len(x) >= npntslim_cosfit)

    # find cosfit fitted velocities
    def cosfunc(x, Amp, phi):
        return Amp * np.cos(1 * x - phi)

    def cosfit(group):
        from scipy.optimize import curve_fit
        fitpars, covmat = curve_fit(cosfunc, np.deg2rad(group.bmazm), group.vel)
        #if fitpars[0] < 0:
        #    fitpars = [-fitpars[0], fitpars[1]+np.pi/2.0]
        perrs = np.sqrt(np.diag(covmat)) 
        df_tmp = pd.DataFrame({'cosfit_vel': fitpars[0], 'cosfit_velerr' : perrs[0],
                               'cosfit_bmazm':np.rad2deg(fitpars[1]) % 360,
                               'cosfit_bmazmerr': np.rad2deg(perrs[1]) % 360},
                               index=group.index)
        N = 1
        return group.join(df_tmp)
    # df that includes cosfit velocities
    df = df.groupby(['latc'], as_index=False).apply(cosfit)
    return df

################# sdvel_cosfit() has to be modified ################


def merge_2losvecs (myMap, df2_griddedvel, velscl=1000., dist=1000.):
    import numpy as np
    import pandas as pd
    rads = [rd for rd in df2_griddedvel.index.levels[0]]
    xp,yp = myMap(0, 90, coords='geo')
    verts_r = [[], []]
    if (not df2_griddedvel.empty) and (df2_griddedvel is not None):
        dfg = df2_griddedvel.groupby(['latc', 'lonc'], as_index=False)
        #df2_merged = 
        groups = []
        for group in dfg:
            group = group[1]
            thetas = []
            lons_rad = group['geolon_rad'].tolist()
            lats_rad = group['geolat_rad'].tolist()
            bazms = group['bmazm'].tolist()
            for ii in range(2):
                xo,yo = myMap(lons_rad[ii], lats_rad[ii], coords='geo')
                the0= np.arctan2(yp-yo,xp-xo)
                theta = the0 - np.deg2rad(bazms[ii])
                thetas.append(theta)
            vel1 = group['vel'].as_matrix()[0]
            vel2 = group['vel'].as_matrix()[1]
            vel1_theta = (np.sign(vel1)+1)/2 * np.pi + thetas[0] 
            vel2_theta = (np.sign(vel2)+1)/2 * np.pi + thetas[1] 
            dtheta = vel2_theta - vel1_theta   # the angle between two gridded LOS vel.
            aa = np.cos(dtheta)
            bb = np.sin(dtheta)
            alpha1 = np.arctan((-aa + (np.abs(vel2)*1.0)/(np.abs(vel1))) / bb)
            if alpha1 > np.pi/2:
                alpha1 += np.pi 
            #theta_r = theta[0] - alpha1
            theta_r = vel1_theta + alpha1
            #vel_r = np.true_divide(vel1, np.cos(alpha1))
            vel_r = vel1 / np.cos(alpha1)
            vel_r = np.abs(vel_r)
            group.loc[:, 'vel_2d'] = np.repeat(vel_r, 2)
            group.loc[:, 'theta_2d'] = np.rad2deg(theta_r)
            groups.append(group)

        df2_merged = pd.concat(groups)
    else:
        df2_merged = None
    return df2_merged

def plot_losvel_az(radars, df_lfitvel, color_list, stime,
                   interval, latc_list=None, 
                   vel_scale=[-150, 150], fig_dir="../plots/scan_plot/"):
    """
    This plots losvel data that go into fitting process.
    This function is called within overlay_2D_sdvel function.
    """

    import datetime as dt
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if latc_list is None:
        latc_list = [x for x in df_lfitvel.latc.unique()]

    npanels = len(latc_list)

    # plot the data that goes into a fitting
    figg1, axx1 = plt.subplots(npanels,1, sharex=True, figsize=(6, 12))
    for r in range(len(radars)):
        latc_list_indv = [x for x in latc_list\
                          if x in df_lfitvel.ix[radars[r]].latc.unique()]
        panel_args = [latc_list.index(x) for x in latc_list_indv] 
        axx1_indv = axx1[panel_args]

        # construct a new dataframe
        df_expr = df_lfitvel.ix[radars[r]][['latc', 'azm_los', 'vel']]
        arr_tmp = df_expr.azm_los.as_matrix() % 360
        indx_tmp = np.where(arr_tmp > 180)
        arr_tmp[indx_tmp] -= 360
        df_expr['azm_los'] = arr_tmp
        df_expr = df_expr.pivot('azm_los', 'latc', 'vel')
        df_expr = df_expr[latc_list_indv]

        # annotate radar names
        figg1.text(0.93, 0.90-r*0.03, radars[r],
                   ha='center',size=10, color=color_list[r])

        # plot losvel vs az for a single radar for certain latc
        try:
            df_expr.plot(subplots=True, ax=axx1_indv, linestyle='',
                         marker='o', markersize=3, mec=color_list[r],
                         mfc=color_list[r], legend=False, grid=False)
        except:
            continue

    figg1.text(0.93, 0.93, 'Radars:',ha='center',size=10)

    df_tmp = df_lfitvel[['latc', 'lfit_azm', 'lfit_vel', 'lfit_vel_err']]
    df_tmp = df_tmp.groupby(['latc'], as_index=False).first()
    df_tmp = df_tmp.set_index(['latc'])
    try:
        df_tmp = df_tmp.loc[latc_list, :]
    except:
        return
   
    df_fit = pd.DataFrame(data=0, index=np.arange(361) - 180, columns=latc_list)
    for l in range(len(latc_list)):
        df_fit[latc_list[l]] = df_tmp.loc[latc_list[l], 'lfit_vel'] *\
                               np.cos(np.radians(df_tmp.loc[latc_list[l], 'lfit_azm'] %\
                               360 - df_fit.index.get_values() % 360))
        axx1[l].set_ylabel(str(latc_list[l]) + '$^\circ$')
        axx1[l].set_xlabel('')
        azm_tmp = df_tmp.loc[latc_list[l], 'lfit_azm'] % 360
        vel_tmp = df_tmp.loc[latc_list[l], 'lfit_vel']
        vel_err_tmp = df_tmp.loc[latc_list[l], 'lfit_vel_err']
        if azm_tmp > 180: azm_tmp -= 360
        axx1[l].plot(azm_tmp, vel_tmp, marker='*',
                     markersize=10, mec='orange', mfc='orange')

#        # set the number of yticks
#        locator = MaxNLocator(nbins=3)
#        axx1[l].yaxis.set_major_locator(locator)
        
        # change yticklabel size
        axx1[l].yaxis.set_tick_params(labelsize=7)

        # mark the peak position
        fsz = 5
        axx1[l].annotate('vel=' + '{0:.01f}'.format(vel_tmp) +\
                         '\nazm=' + '{0:.01f}'.format(azm_tmp) +'$^\circ$' +\
                         '\nvel_std=' + '{0:.01f}'.format(vel_err_tmp), xy = (1.02, 0.20),
                         xycoords='axes fraction', horizontalalignment='left',
                         verticalalignment='bottom', fontsize=fsz)
    
    df_fit.plot(subplots=True, ax=axx1, linewidth=1.0,
                marker='.', markersize=1, mec='k', mfc='k',
                legend=False, grid=True, ylim=vel_scale)
    axx1[-1].set_xlabel(r'Azimuth [$^{\circ}$]')
    axx1[0].set_title(stime.strftime('%Y/%m/%d            %H:%M') + ' - ' +\
                      (stime+dt.timedelta(seconds=interval)).strftime('%H:%M  UT'),
                      ha='center',size=12,weight=550)
    
    #handle the outputs
    figg1.savefig(fig_dir + 'panelplot_' + "_".join(radars)  + "_" +\
                  stime.strftime("%Y%m%d.%H%M") + "_to_" +\
                  (stime+dt.timedelta(seconds=interval)).strftime("%Y%m%d.%H%M")+\
                  '.png', dpi=300, bbox_inches="tight")

    plt.close(figg1)

###################################################################


#sd        = ['ksr','kod','pgr','sas','kap','gbr','pyk','han','sto']; These are the high-latitude superdarn radars plotted in blue.
#sd_nbeams = [   16,   16,   16,   16,   16,   16,   16,   16,   16]
#;sdm= [[b1], [b7], [b12],[b6], [b7], [b7], [b5], [b9]]  ;Camping Beams
#pd        = ['inv','rkn','cly'] ;These are the polar darn radars plotted in green.

#rads = ["rkn", "inv", "cly"]
##rads = ["rkn"]
##rads = ["inv"]
##stime = dt.datetime(2014,9,12,19,00)
#import datetime as dt
#stime = dt.datetime(2014,9,13,19,19)
#interval=1*60
#fileType = 'fitex'; params = 'velocity'
#channel=None; filtered=False
#coords = 'mlt'
#npntslim_lfit = 5
#from davitpy import utils
#myMap = utils.mapObj(coords=coords,projection='stere', width=10.0**3, 
#                         height=10.0**3, lat_0=90, lon_0=180,
#                         datetime = stime)
#
#df_griddedvel = grid_sddata(rads,rads_data=None, fovs=None,sites=None, min_range_lim= 450, npnts_minlim=3,
#        stime=stime,interval=interval,fileType=fileType,filtered=filtered,channel=channel,coords=coords, 
#        lat_min=60,lat_max=90,dlat=1,half_dlat_offset=False)
#df_lfitvel =  sdvel_lfit(myMap, df_griddedvel, npntslim_lfit=npntslim_lfit)
#df_azm = calc_azm(myMap, df_griddedvel)
#df2_griddedvel = df_griddedvel.groupby(['latc', 'lonc'], as_index=False).filter(lambda x: len(x)==2)
#df2_merged = merge_2losvecs(myMap, df2_griddedvel)
#N = 1

