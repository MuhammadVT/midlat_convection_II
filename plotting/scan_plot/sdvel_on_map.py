class sdvel_on_map(object):

    def __init__(self, ax, rads, stime,
                 interval=120,
                 map_lat0=90, map_lon0=0,
                 map_width=50*111e3, 
                 map_height=50*111e3, 
                 map_resolution='l', 
		 rot_map=0.0,
		 coords="mlt",
		 channel=None,
		 fileType="fitacf"):

        from davitpy import utils

	self.stime = stime
	self.interval = interval
	self.etime = stime+dt.timedelta(seconds=interval)
	self.rads = rads
	self.coords = coords
        self.fileType = fileType
        self.channel = channel
        self.ax = ax

	# load the data, create sites and fovs for rads
	self.data = self._load_sddata()

        # Create a map
        map_obj = utils.mapObj(coords=coords, projection='stere',
			       width=map_width, height=map_height,
			       lat_0=map_lat0, lon_0=map_lon0,
			       resolution=map_resolution,
			       datetime=stime, showCoords=True)
    
	# to display the map with 12 MLT located in the top side
	llcrnrlon, llcrnrlat = map_obj.llcrnrlon, map_obj.llcrnrlat
	urcrnrlon, urcrnrlat = map_obj.urcrnrlon, map_obj.urcrnrlat

	# clear the current axes
	plt.cla()
    
        # rotate the map
        map_lon0 += rot_map
    
        # draw the actual map we want
        self.map_obj = utils.mapObj(ax=ax, coords=coords, projection='stere',
				    lat_0=map_lat0, lon_0=map_lon0,
				    llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
				    urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
				    datetime=stime)
	
    def _load_sddata(self):
        """ Loads the data of interest"""

	from davitpy import pydarn

        # open the data files
        myPtrs = []
        for i in range(len(self.rads)):
	    if self.channel:
		channel = self.channel[i]
            else:
                channel = self.channel
            f = pydarn.sdio.radDataOpen(self.stime, self.rads[i],
					self.etime,
					fileType=self.fileType,
				 	channel=channel)
            if f is not None :
                myPtrs.append(f)
            else:
                myPtrs.append(None)
    
        #go through all open files
        sites, fovs, nrangs=[],[],[]
	rads_data = [[] for i in range(len(myPtrs))]
        for i in range(len(myPtrs)):
            myBeam = pydarn.sdio.radDataReadRec(myPtrs[i])

            #check that we have good data at this time
            if (myPtrs[i] is None) or (myBeam is None):
                myPtrs[i].close()
                rads_data[i] = None
                fovs.append(None)
                sites.append(None)
                nrangs.append(None)
                continue
	    else:
		nrangs.append(myBeam.prm.nrang)
                # create site object
		t=myPtrs[i].sTime
		site = pydarn.radar.site(radId=myBeam.stid,dt=t)
		sites.append(site)
	
		# create fov object
		myFov = pydarn.radar.radFov.fov(site=site, rsep=myBeam.prm.rsep,\
						ngates=myBeam.prm.nrang+1,
						nbeams=site.maxbeam, coords=self.coords,
						date_time=t)
		fovs.append(myFov)

            # read until we reach the end of the time window
            while(myBeam is not None):
		if (myBeam.time > myPtrs[i].eTime): 
		    break
		if (myBeam.time >= myPtrs[i].sTime):
		    rads_data[i].append(myBeam)
                myBeam = myPtrs[i].readRec()

	    if rads_data[i] == []:
		rads_data[i] = None

	self.sites = sites
	self.fovs = fovs
        return rads_data
    def show_map(self):
        import matplotlib.pyplot as plt
        plt.show()

    def overlay_fov():
	pass

    def overlay_losvel(myData,myMap,param,coords='geo',gsct=0,site=None,\
                                    fov=None,fill=True,velscl=1000.,dist=1000.,
                                    cmap=None,norm=None,alpha=1):
    
        from davitpy import pydarn
        if(site is None):
            site = pydarn.radar.site(radId=myData[0].stid, dt=myData[0].time)
        if(fov is None):
            fov = pydarn.radar.radFov.fov(site=site, rsep=myData[0].prm.rsep,
                                          ngates=myData[0].prm.nrang+1,
                                          nbeams= site.maxbeam, coords=coords,
                                          date_time=myData[0].time)
    
        if(isinstance(myData,pydarn.sdio.beamData)): myData = [myData]
    
        gs_flg,lines = [],[]
        if fill: verts,intensities = [],[]
        else: verts,intensities = [[],[]],[[],[]]
    
        #loop through gates with scatter
        for myBeam in myData:
            for k in range(0,len(myBeam.fit.slist)):
                if myBeam.fit.slist[k] not in fov.gates: continue
                r = myBeam.fit.slist[k]
    
                if fill:
                    x1,y1 = myMap(fov.lonFull[myBeam.bmnum,r],fov.latFull[myBeam.bmnum,r])
                    x2,y2 = myMap(fov.lonFull[myBeam.bmnum,r+1],fov.latFull[myBeam.bmnum,r+1])
                    x3,y3 = myMap(fov.lonFull[myBeam.bmnum+1,r+1],fov.latFull[myBeam.bmnum+1,r+1])
                    x4,y4 = myMap(fov.lonFull[myBeam.bmnum+1,r],fov.latFull[myBeam.bmnum+1,r])
    
                    #save the polygon vertices
                    verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))
    
                    #save the param to use as a color scale
                    if(param == 'velocity'): intensities.append(myBeam.fit.v[k])
    
                else:
                    x1,y1 = myMap(fov.lonCenter[myBeam.bmnum,r],fov.latCenter[myBeam.bmnum,r])
                    verts[0].append(x1)
                    verts[1].append(y1)
    
                    x2,y2 = myMap(fov.lonCenter[myBeam.bmnum,r+1],fov.latCenter[myBeam.bmnum,r+1])
    
                    theta = math.atan2(y2-y1,x2-x1)
    
                    x2,y2 = x1+myBeam.fit.v[k]/velscl*(-1.0)*math.cos(theta)*dist,y1+myBeam.fit.v[k]/velscl*(-1.0)*math.sin(theta)*dist
    
                    lines.append(((x1,y1),(x2,y2)))
                    #save the param to use as a color scale
                    if(param == 'velocity'): intensities[0].append(myBeam.fit.v[k])
                if(gsct): gs_flg.append(myBeam.fit.gflg[k])
    
    
        #do the actual overlay
        if(fill):
            #if we have data
            if(verts != []):
                if(gsct == 0):
                    inx = np.arange(len(verts))
                else:
                    inx = np.where(np.array(gs_flg)==0)
                    x = PolyCollection(np.array(verts)[np.where(np.array(gs_flg)==1)],
                        facecolors='.3',linewidths=0,zorder=5,alpha=alpha)
                    myMap.ax.add_collection(x, autolim=True)
    
                pcoll = PolyCollection(np.array(verts)[inx],
                    edgecolors='face',linewidths=0,closed=False,zorder=4,
                    alpha=alpha,cmap=cmap,norm=norm)
                #set color array to intensities
                pcoll.set_array(np.array(intensities)[inx])
                myMap.ax.add_collection(pcoll, autolim=True)
                return intensities,pcoll
        else:
            #if we have data
            if(verts != [[],[]]):
                if(gsct == 0):
                    inx = np.arange(len(verts[0]))
                else:
                    inx = np.where(np.array(gs_flg)==0)
                    #plot the ground scatter as open circles
                    x = myMap.ax.scatter(np.array(verts[0])[np.where(np.array(gs_flg)==1)],\
                            np.array(verts[1])[np.where(np.array(gs_flg)==1)],\
                            #s=.1*np.array(intensities[1])[np.where(np.array(gs_flg)==1)],\
                            s=3.0,\
                            zorder=6,marker='o',linewidths=.5,facecolors='w',edgecolors='k')
                    myMap.ax.add_collection(x, autolim=True)
    
                #plot the i-s as filled circles
                ccoll = myMap.ax.scatter(np.array(verts[0])[inx],np.array(verts[1])[inx],
                                #s=.1*np.array(intensities[1])[inx],zorder=10,marker='o',
                                s=3.0,zorder=10,marker='o', c=np.abs(np.array(intensities[0])[inx]),
                                linewidths=.5, edgecolors='face',cmap=cmap,norm=norm)
    
                #set color array to intensities
                #ccoll.set_array(np.array(intensities[0])[inx])
                myMap.ax.add_collection(ccoll)
                #plot the velocity vectors
                lcoll = LineCollection(np.array(lines)[inx],linewidths=.5,zorder=12,cmap=cmap,norm=norm)
                lcoll.set_array(np.abs(np.array(intensities[0])[inx]))
                myMap.ax.add_collection(lcoll)
    
                return intensities,lcoll


    def griddedVel(self, rads, myData, fovs,lat_min=60,lat_max=90,dlat=1, range_minlim= 450, npnts_minlim=3,
            half_dlat_offset=False):
    
        '''
        "half_dlat_offset=False" implements NINT[360 sin(theta)] at theta = 89, 88, ... colatitude
        "half_dlat_offset=True" implements NINT[360 sin(theta)] at theta = 89.5, 88.5, ... colatitude
        '''
    
        lats = [x + 0.5*dlat for x in range(lat_min,lat_max,dlat)]
        if half_dlat_offset:
            nlons = [round(360 * np.sin(np.deg2rad(90-lat))) for lat in lats]
        else:
            nlons = [round(360 * np.sin(np.deg2rad(90-(lat-0.5*dlat)))) for lat in lats]
        dlons = [360./nn for nn in nlons]
    
        # lat and lon bins
        lat_bins = [x for x in np.arange(lat_min,lat_max+dlat,dlat)]
        lon_bins = []
        lonss = []      # list of lists of lons
        for i in range(len(lats)):
            lon_tmp = [ item*dlons[i] for item in np.arange(0.5, nlons[i]+0.5) ]
            lonss.append(lon_tmp)
            lon_tmp = [ item*dlons[i] for item in np.arange(nlons[i]) ]
            lon_tmp.append(360)
            lon_bins.append(lon_tmp)
    
        df_lst = []
        rads_lft = []
        for i in range(len(myData)):
            scan = myData[i]
            if scan is None:
                continue
            fov = fovs[i]
            # gridded parameters
            gvels, glats, glons, gbmazms = [], [], [], []
            for myBeam in scan:
                if (len(myBeam.fit.slist)==0) or (myBeam is None): continue
                for k in range(0,len(myBeam.fit.slist)):
                    if myBeam.fit.slist[k] not in fov.gates:
                        continue
                    if (myBeam.fit.slist[k] * myBeam.prm.rsep) < range_minlim:
                        continue
                    if (myBeam.fit.gflg[k]):     # filter out ground scatter
                        continue
                    r = myBeam.fit.slist[k]
                    ilon = fov.lonCenter[myBeam.bmnum,r]
                    ilat = fov.latCenter[myBeam.bmnum,r]
                    ivel = myBeam.fit.v[k]
                    bmazm = myBeam.prm.bmazm
                    indx_lat = np.digitize([ilat], lat_bins)
                    glat = lats[indx_lat - 1]
                    indx_lon = np.digitize([ilon % 360], lon_bins[indx_lat-1])
                    glon = lonss[indx_lat-1][indx_lon-1]
                    glats.append(glat)
                    glons.append(glon)
                    gvels.append(ivel)
                    gbmazms.append(bmazm)
    
            rads_lft.append(rads[i])
            columns = ['latc', 'lonc', 'vel_'+rads_lft[i], 'bmazm_'+rads_lft[i]]
            df_tmp = pd.DataFrame(zip(glats, glons, gvels, gbmazms), columns=columns)
            df_tmp = df_tmp.sort(['latc', 'lonc'])
            df_tmp = df_tmp.groupby(['latc', 'lonc'], as_index=False).filter(lambda x: len(x)>= npnts_minlim)
            #df_tmp = df_tmp.groupby(['latc', 'lonc', 'bmazm_'+rads_lft[i]], as_index=False).median()
            df_tmp = df_tmp.groupby(['latc', 'lonc'], as_index=False).median()
            df_lst.append(df_tmp)
        if len(rads_lft) == 2:     # works for overlapped areas covered by a pair of radar
            df = pd.merge(df_lst[0], df_lst[1], on=['latc', 'lonc'], how='inner')
        elif len(rads_lft) == 1:   # works for a single radar
            df = df_lst[0]
        else:
            df = None
        return df



    def overlay_grids(myMap,lat_min=60,lat_max=90,dlat=1,
            llcrnrlat=None,urcrnrlat=None,half_dlat_offset=True,date_time = None):
        """
        Note
        ----
        half_dlat_offset=False implements NINT[360 sin(theta)] at theta = 89, 88, ... colatitude
            and the center latitudes are 89.5 ,88.5, etc.
        half_dlat_offset=True implements NINT[360 sin(theta)] at theta = 89.5, 88.5, ... colatitude
    
        """
    
        import numpy as np
        import datetime as dt
    
        if not date_time:
            date_time = dt.now()
        if llcrnrlat==None or urcrnrlat==None:
            lats = [x + 0.5*dlat for x in np.arange(lat_min,lat_max,dlat)]
        else:
            lat_min = np.floor(min(llcrnrlat, urcrnrlat))
            lats = [x + 0.5*dlat for x in np.arange(lat_min,lat_max,dlat)]
        if half_dlat_offset:
            nlons = [round(360 * np.sin(np.deg2rad(90-lat))) for lat in lats]
        else:
            nlons = [round(360 * np.sin(np.deg2rad(90-(lat-0.5*dlat)))) for lat in lats]
        dlons = [360./nn for nn in nlons]
    
        # flatting all lats and lons 
        lons_all = np.array([])
        lons_all_E = np.array([])
        lons_all_W = np.array([])
        lats_all = np.array([])
        lats_all_N = np.array([])
        lats_all_S = np.array([])
        for i in range(len(lats)):
            lons = [ item*dlons[i] for item in np.arange(0.5, (nlons[i]+0.5)) ]
            lons_E = [ item*dlons[i] for item in np.arange(1, (nlons[i]+1)) ]
            lons_W = [ item*dlons[i] for item in np.arange(0, (nlons[i])) ]
            lons_all = np.append(lons_all, lons)
            lons_all_E = np.append(lons_all_E, lons_E)
            lons_all_W = np.append(lons_all_W, lons_W)
    
            lats_all = np.append(lats_all, np.repeat(lats[i], nlons[i]))
            lats_all_N = np.append(lats_all_N, np.repeat((lats[i]+dlat*0.5), nlons[i]))
            lats_all_S = np.append(lats_all_S, np.repeat((lats[i]-dlat*0.5), nlons[i]))
        # plot grid
        x1,y1 = myMap(lons_all_W,lats_all_S)
        x2,y2 = myMap(lons_all_W,lats_all_N)
        x3,y3 = myMap(lons_all_E,lats_all_N)
        x4,y4 = myMap(lons_all_E,lats_all_S)
        verts = zip(zip(x1,y1), zip(x2,y2), zip(x3,y3), zip(x4,y4))
        pcoll = PolyCollection(np.array(verts), linewidth=0.07, facecolor='', zorder=5, clip_on=True)
        #myMap.ax.add_collection(pcoll)
        plt.gca().add_collection(pcoll)

    def overlay_griddedVel(rads, scans, fovs, sites, myMap, myFig,
            coords='geo',cmap=None,norm=None, velscl=1000.0, dist=1000.0,
            lat_min=60,lat_max=90,dlat=1,half_dlat_offset=False,
            show_gridded_losvel=False, show_overlapped_gridded_losvel_only=False):
    
        verts = [[], []]
        intensities = []
        lines = []
        if show_gridded_losvel:
            if  show_overlapped_gridded_losvel_only: 
                df = griddedVel(rads, scans, fovs)
                for i in range(len(rads)): 
                    site = sites[i]
                    lon_rad = site.geolon
                    lat_rad = site.geolat
                    xo,yo = myMap(lon_rad, lat_rad, coords='geo')
                    xp,yp = myMap(0, 90, coords='geo')
                    theta = np.arctan2(yp-yo,xp-xo)
                    theta = theta - np.deg2rad(df['bmazm_'+rads[i]].tolist())
    
                    x1,y1 = myMap(df['lonc'].tolist(), df['latc'].tolist())
                    verts[0].extend(x1)
                    verts[1].extend(y1)
                    x1 = np.array(x1)
                    y1 = np.array(y1)
                    x2 = x1+np.array(df['vel_'+rads[i]].tolist())/velscl*(-1.0)*np.cos(theta)*dist
                    y2 = y1+np.array(df['vel_'+rads[i]].tolist())/velscl*(-1.0)*np.sin(theta)*dist
    
                    lines.extend(zip(zip(x1,y1),zip(x2,y2)))
                    #save the param to use as a color scale
                    intensities.extend(df['vel_'+rads[i]].tolist())
    
    
                #do the actual overlay
                #plot the i-s as filled circles
                ccoll = myFig.gca().scatter(np.array(verts[0]),np.array(verts[1]),
                                s=3.0,zorder=10,marker='o', c=np.abs(np.array(intensities)),
                                #s=3.0,zorder=10,marker='o', color='k',
                                linewidths=.5, edgecolors='face',cmap=cmap,norm=norm)
                                #linewidths=.5, edgecolors='face')
                myFig.gca().add_collection(ccoll)
                #plot the velocity vectors
                lcoll = LineCollection(np.array(lines),linewidths=0.5,zorder=12,cmap=cmap,norm=norm)
                lcoll.set_array(np.abs(np.array(intensities)))
                myFig.gca().add_collection(lcoll)
            else:
                for i in range(len(rads)):
                    df = griddedVel([rads[i]], [scans[i]], [fovs[i]])
                    site = sites[i]
                    lon_rad = site.geolon
                    lat_rad = site.geolat
                    xo,yo = myMap(lon_rad, lat_rad, coords='geo')
                    xp,yp = myMap(0, 90, coords='geo')
                    theta = np.arctan2(yp-yo,xp-xo)
                    theta = theta - np.deg2rad(df['bmazm_'+rads[i]].tolist())
    
                    x1,y1 = myMap(df['lonc'].tolist(), df['latc'].tolist())
                    verts[0].extend(x1)
                    verts[1].extend(y1)
                    x1 = np.array(x1)
                    y1 = np.array(y1)
    
                    x2 = x1+np.array(df['vel_'+rads[i]].tolist())/velscl*(-1.0)*np.cos(theta)*dist
                    y2 = y1+np.array(df['vel_'+rads[i]].tolist())/velscl*(-1.0)*np.sin(theta)*dist
    
                    lines.extend(zip(zip(x1,y1),zip(x2,y2)))
                    #save the param to use as a color scale
                    intensities.extend(df['vel_'+rads[i]].tolist())
    
                #do the actual overlay
                #plot the i-s as filled circles
                ccoll = myFig.gca().scatter(np.array(verts[0]),np.array(verts[1]),
                                #s=.1*np.array(intensities[1])[inx],zorder=10,marker='o',
                                s=3.0,zorder=10,marker='o', c=np.abs(np.array(intensities)),
                                linewidths=.5, edgecolors='face',cmap=cmap,norm=norm)
    
                myFig.gca().add_collection(ccoll)
                #plot the velocity vectors
                lcoll = LineCollection(np.array(lines),linewidths=0.5,zorder=12,cmap=cmap,norm=norm)
                lcoll.set_array(np.abs(np.array(intensities)))
                myFig.gca().add_collection(lcoll)
        # resolved 2D velocity 
        thetas = []   # the collection of angles of two pair radars' gridded LOS vel.
        df = griddedVel(rads, scans, fovs)
        if (df is not None):
            dfg = df.groupby(['latc', 'lonc'], as_index=False).filter(lambda x: len(x) == 2)
        if (df is not None) and (not dfg.empty):
            for i in range(len(rads)):
                site = sites[i]
                lon_rad = site.geolon
                lat_rad = site.geolat
                xo,yo = myMap(lon_rad, lat_rad, coords='geo')
                xp,yp = myMap(0, 90, coords='geo')
                theta = np.arctan2(yp-yo,xp-xo)
                theta = theta - np.deg2rad(df['bmazm_'+rads[i]].tolist())
                thetas.append(theta)
    
            verts_r = [[], []]
            vel1 = df['vel_'+rads[0]].as_matrix()
            vel2 = df['vel_'+rads[1]].as_matrix()
            vel1_theta = (np.sign(vel1)+1)/2 * np.pi + thetas[0]
            vel2_theta = (np.sign(vel2)+1)/2 * np.pi + thetas[1]
            dtheta = vel2_theta - vel1_theta   # the angle between two gridded LOS vel.
            aa = np.cos(dtheta)
            bb = np.sin(dtheta)
            alpha1 = np.arctan((-aa + (np.abs(vel2)*1.0)/(np.abs(vel1))) / bb)
            alpha1[np.where(alpha1 > np.pi/2)] += np.pi
            #theta_r = theta[0] - alpha1
            theta_r = vel1_theta + alpha1
            #vels_r = np.true_divide(vel1, np.cos(alpha1))
            vels_r = vel1 / np.cos(alpha1)
            vels_r = np.abs(vels_r)
    
            xr1,yr1 = myMap(df['lonc'].tolist(), df['latc'].tolist())
            verts_r[0].extend(xr1)
            verts_r[1].extend(yr1)
            xr1 = np.array(xr1)
            yr1 = np.array(yr1)
    
            xr2 = xr1+vels_r/velscl*(+1.0)*np.cos(theta_r)*dist
            yr2 = yr1+vels_r/velscl*(+1.0)*np.sin(theta_r)*dist
    
            lines_r = zip(zip(xr1,yr1),zip(xr2,yr2))
            #save the param to use as a color scale
            intensities_r = vels_r.tolist()
            ccoll_r = myFig.gca().scatter(np.array(verts_r[0]),np.array(verts_r[1]),
                            s=3.0,zorder=11,marker='*', c=np.array(intensities_r),
                            linewidths=.5, edgecolors='face',cmap=cmap,norm=norm)
            #set color array to intensities
            lcoll_r = LineCollection(np.array(lines_r),linewidths=1.0,zorder=12,cmap=cmap,norm=norm)
            lcoll_r.set_array(np.array(intensities_r))
            myFig.gca().add_collection(ccoll_r)
            myFig.gca().add_collection(lcoll_r)
    
            return intensities_r,lcoll_r
            #return intensities,lcoll
        else:
            return [None, None]

    def overlay_2D_sdvel(stime, rads, rads_data, fovs, sites, myMap,
            coords='geo',npnts_minlim=3,npntslim_lfit=5,
            interval=1*60,fileType='fitex',filtered=False,channel=None,
            lat_min=60,lat_max=90,dlat=1,half_dlat_offset=False,
            cmap=None,norm=None, velscl=1000.0, dist=1000.0, min_range_lim=0, max_lfit_vel_lim=None,
            png=False, overlay_gridded_losvel=False,
            gridded_losvel_only=False,all_lfitvel=False,hybrid_2Dvel=True,
            nazmslim_pr_grid=1, fitting_diagnostic_plot=True):
    
        df_griddedvel = grid_sddata(rads,rads_data=rads_data, fovs=fovs,sites=sites, min_range_lim= min_range_lim, npnts_minlim=npnts_minlim,
                stime=stime,interval=interval,fileType=fileType,filtered=filtered,channel=channel,coords=coords,
                lat_min=lat_min,lat_max=lat_max,dlat=dlat,half_dlat_offset=half_dlat_offset)
    
        if df_griddedvel is not None:
    
            # show the supdardarn gridded los velocities
            if overlay_gridded_losvel:
                verts = [[], []]
                intensities = []
                lines = []
                xp, yp = myMap(0, 90, coords='geo')    # geographic pole position on the map
                xo, yo = myMap(df_griddedvel.geolon_rad.tolist(), df_griddedvel.geolat_rad.tolist(), coords='geo')
                the0 = np.arctan2(yp-np.array(yo),xp-np.array(xo))
                # NOTE: In the angle calculation below, the better way is to implement the spherical trigonometry.
                theta = the0 - np.deg2rad(df_griddedvel['bmazm'].tolist())
                x1,y1 = myMap(df_griddedvel['lonc'].tolist(), df_griddedvel['latc'].tolist())
                verts[0].extend(x1)
                verts[1].extend(y1)
                x1 = np.array(x1)
                y1 = np.array(y1)
    
                x2 = x1+np.array(df_griddedvel['vel'].tolist())/velscl*(-1.0)*np.cos(theta)*dist
                y2 = y1+np.array(df_griddedvel['vel'].tolist())/velscl*(-1.0)*np.sin(theta)*dist
    
                lines.extend(zip(zip(x1,y1),zip(x2,y2)))
                #save the param to use as a color scale
                intensities.extend(df_griddedvel['vel'].tolist())
    
                #do the actual overlay
                ccoll = myMap.ax.scatter(np.array(verts[0]),np.array(verts[1]),
                                #s=3.0,zorder=10,marker='o', c=np.abs(np.array(intensities)),
                                s=3.0,zorder=10,marker='o', color='grey',
                                linewidths=.5, edgecolors='face',cmap=cmap,norm=norm)
    
                lcoll = LineCollection(np.array(lines),linewidths=0.5,zorder=12,color='grey', cmap=cmap,norm=norm)
                #lcoll = LineCollection(np.array(lines),linewidths=0.5,zorder=12,cmap=cmap,norm=norm)
                #lcoll.set_array(np.abs(np.array(intensities)))
                if gridded_losvel_only:
                    ccoll_grey = myMap.ax.scatter(np.array(verts[0]),np.array(verts[1]),
                                    s=3.0,zorder=10,marker='o', color='grey',
                                    linewidths=.5, edgecolors='face',cmap=cmap,norm=norm)
                    lcoll_grey = LineCollection(np.array(lines),linewidths=0.5,zorder=12,color='grey', cmap=cmap,norm=norm)
                    myMap.ax.add_collection(ccoll_grey)
                    myMap.ax.add_collection(lcoll_grey)
                else:
                    myMap.ax.add_collection(ccoll)
                    myMap.ax.add_collection(lcoll)
    
            # show the supdardarn gridded los velocities only if gridded_losvel_only=True
            # this way fitted velocites will not be shown
            if not gridded_losvel_only:
                df_lfitvel =  sdvel_lfit(myMap, df_griddedvel, npntslim_lfit=npntslim_lfit)
    
                radars = [rd for rd in df_lfitvel.index.levels[0]]
                color_list = ['r', 'b', 'g', 'c', 'm', 'k']
    
                if all_lfitvel and (df_lfitvel is not None):
    
                    # filter out the lfit velocity that exceedes max_lfit_vel_lim
                    if max_lfit_vel_lim is not None:
                        df_lfitvel = df_lfitvel.where(df_lfitvel.lfit_vel < max_lfit_vel_lim).dropna()
    
                    verts = [[], []]
                    intensities = []
                    lines = []
                    xp, yp = myMap(0, 90)    # North pole in the myMap.coords system
                    xo, yo = myMap(df_lfitvel.lonc.tolist(), df_lfitvel.latc.tolist())
                    the0 = np.arctan2(yp-np.array(yo),xp-np.array(xo))
                    theta = the0 - np.deg2rad(df_lfitvel['lfit_azm'].tolist())
                    x1,y1 = myMap(df_lfitvel['lonc'].tolist(), df_lfitvel['latc'].tolist())
                    verts[0].extend(x1)
                    verts[1].extend(y1)
                    x1 = np.array(x1)
                    y1 = np.array(y1)
    
                    x2 = x1+np.array(df_lfitvel['lfit_vel'].tolist())/velscl*(-1.0)*np.cos(theta)*dist
                    y2 = y1+np.array(df_lfitvel['lfit_vel'].tolist())/velscl*(-1.0)*np.sin(theta)*dist
    
                    lines.extend(zip(zip(x1,y1),zip(x2,y2)))
                    #save the param to use as a color scale
                    intensities.extend(df_lfitvel['lfit_vel'].tolist())
    
                    #do the actual overlay
                    #plot the i-s as filled circles
                    ccoll = myMap.ax.scatter(np.array(verts[0]),np.array(verts[1]),
                                    #s=.1*np.array(intensities[1])[inx],zorder=10,marker='o',
                                    s=3.0,zorder=10,marker='o', c=np.abs(np.array(intensities)),
                                    linewidths=.5, edgecolors='face',cmap=cmap,norm=norm)
    
                    myMap.ax.add_collection(ccoll)
                    #plot the velocity vectors
                    lcoll = LineCollection(np.array(lines),linewidths=0.5,zorder=12,cmap=cmap,norm=norm)
                    lcoll.set_array(np.abs(np.array(intensities)))
                    myMap.ax.add_collection(lcoll)
                df1_griddedvel = df_griddedvel.groupby(['latc', 'lonc'], as_index=False).filter(lambda x: len(x)==1)
                df2_griddedvel = df_griddedvel.groupby(['latc', 'lonc'], as_index=False).filter(lambda x: len(x)==2)
                df1_lfitvel = df_lfitvel.groupby(['latc', 'lonc']).filter(lambda x: len(x)==1)
                if hybrid_2Dvel:
                    verts = [[], []]
                    intensities = []
                    lines = []
                    if nazmslim_pr_grid < 2:
                        # for df1_lfitvel
                        if (not df1_lfitvel.empty):
                            xp, yp = myMap(0, 90)    # geomagnetic pole position on the map
                            xo, yo = myMap(df1_lfitvel.lonc.tolist(), df1_lfitvel.latc.tolist())
                            the0 = np.arctan2(yp-np.array(yo),xp-np.array(xo))
                            theta = the0 - np.deg2rad(df1_lfitvel['lfit_azm'].tolist())
                            x1,y1 = myMap(df1_lfitvel['lonc'].tolist(), df1_lfitvel['latc'].tolist())
                            verts[0].extend(x1)
                            verts[1].extend(y1)
                            x1 = np.array(x1)
                            y1 = np.array(y1)
                            x2 = x1+np.array(df1_lfitvel['lfit_vel'].tolist())/velscl*(-1.0)*np.cos(theta)*dist
                            y2 = y1+np.array(df1_lfitvel['lfit_vel'].tolist())/velscl*(-1.0)*np.sin(theta)*dist
    
                            lines.extend(zip(zip(x1,y1),zip(x2,y2)))
                            #save the param to use as a color scale
                            intensities.extend(df1_lfitvel['lfit_vel'].tolist())
    
                    # for df2_merged
                    df2_merged = merge_2losvecs(myMap, df2_griddedvel)
                    if df2_merged is not None:
                        xr1,yr1 = myMap(df2_merged['lonc'].tolist(), df2_merged['latc'].tolist())
                        verts[0].extend(xr1)
                        verts[1].extend(yr1)
                        xr1 = np.array(xr1)
                        yr1 = np.array(yr1)
                        vel_r = df2_merged['vel_2d'].as_matrix()
                        theta_r = np.deg2rad(df2_merged['theta_2d'].as_matrix())
                        xr2 = xr1+vel_r/velscl*(+1.0)*np.cos(theta_r)*dist
                        yr2 = yr1+vel_r/velscl*(+1.0)*np.sin(theta_r)*dist
    
                        lines.extend(zip(zip(xr1,yr1),zip(xr2,yr2)))
                        #save the param to use as a color scale intensities.extend(vel_r.tolist())
                        intensities.extend(df2_merged['vel_2d'].tolist())
    
                    # for dfge3_lfitvel
                    dfge3_griddedvel = df_griddedvel.groupby(['latc', 'lonc'], as_index=False).filter(lambda x: len(x)>=3)
                    dfgge3 = dfge3_griddedvel.groupby(['latc', 'lonc'], as_index=False)
                    groups_lfit=[]
                    for indx, group in dfgge3:
                        group_lfit = sdvel_lfit(myMap, group, npntslim_lfit=3)
                        if group_lfit is not None:
                            groups_lfit.append(group_lfit)
                    if (not groups_lfit==[]):
                        dfge3_lfitvel = pd.concat(groups_lfit)
                        xp, yp = myMap(0, 90)    # geomagnetic pole position on the map
                        xo, yo = myMap(dfge3_lfitvel.lonc.tolist(), dfge3_lfitvel.latc.tolist())
                        the0 = np.arctan2(yp-np.array(yo),xp-np.array(xo))
                        theta = the0 - np.deg2rad(dfge3_lfitvel['lfit_azm'].tolist())
                        x1,y1 = myMap(dfge3_lfitvel['lonc'].tolist(), dfge3_lfitvel['latc'].tolist())
                        verts[0].extend(x1)
                        verts[1].extend(y1)
                        x1 = np.array(x1)
                        y1 = np.array(y1)
                        x2 = x1+np.array(dfge3_lfitvel['lfit_vel'].tolist())/velscl*(-1.0)*np.cos(theta)*dist
                        y2 = y1+np.array(dfge3_lfitvel['lfit_vel'].tolist())/velscl*(-1.0)*np.sin(theta)*dist
    
                        lines.extend(zip(zip(x1,y1),zip(x2,y2)))
                        #save the param to use as a color scale
                        intensities.extend(dfge3_lfitvel['lfit_vel'].tolist())
    
    
                    #do the actual overlay
                    #plot the i-s as filled circles
                    ccoll = myMap.ax.scatter(np.array(verts[0]),np.array(verts[1]),
                                    #s=.1*np.array(intensities[1])[inx],zorder=10,marker='o',
                                    s=3.0,zorder=10,marker='o', c=np.abs(np.array(intensities)),
                                    linewidths=.5, edgecolors='face',cmap=cmap,norm=norm)
    
                    myMap.ax.add_collection(ccoll)
                    #plot the velocity vectors
                    lcoll = LineCollection(np.array(lines),linewidths=0.5,zorder=12,cmap=cmap,norm=norm)
                    lcoll.set_array(np.abs(np.array(intensities)))
                    myMap.ax.add_collection(lcoll)
    
                if fitting_diagnostic_plot:
                    latc_list = [x + 0.5 for x in range(52, 59)]
                    #latc_list=None
                    plot_az_losvel(radars, df_lfitvel, color_list, stime, interval,
                                   latc_list=latc_list)
        else:
            intensities, lcoll = None, None
        return intensities,lcoll

    def add_cbar(fig, coll, bounds, label="Velocity [m/s]", cax=None,
                 title_size=14, ytick_label_size=12):
    
        # add color bar
        if cax:
            cbar=fig.colorbar(coll, cax=cax, orientation="vertical",
                              boundaries=bounds, drawedges=False)
        else:
            cbar=fig.colorbar(coll, orientation="vertical", shrink=.65,
                              boundaries=bounds, drawedges=False)
    
        #define the colorbar labels
        l = []
        for i in range(0,len(bounds)):
            if i == 0 or i == len(bounds)-1:
                l.append(' ')
                continue
            l.append(str(int(bounds[i])))
        cbar.ax.set_yticklabels(l)
        cbar.ax.tick_params(axis='y',direction='out')
        cbar.set_label(label)
    
        #set colorbar ticklabel size
        for ti in cbar.ax.get_yticklabels():
            ti.set_fontsize(ytick_label_size)
        cbar.set_label('Velocity [m/s]',size=title_size)
        cbar.extend='max'

if __name__ == "__main__":
    import datetime as dt
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8,6))
    stime = dt.datetime(2012, 11, 7, 4, 0)
    interval = 2*60
    coords = "mlt"
    rads = ["cve", "cvw"]
    obj = sdvel_on_map(ax, rads, stime, interval=interval,
                       map_lat0=50, map_lon0=0,
                       map_width=50*111e3, 
                       map_height=50*111e3, 
                       map_resolution='l', 
                       coords=coords,
                       channel=None,
                       fileType="fitacf")

