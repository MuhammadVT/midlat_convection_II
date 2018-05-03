import matplotlib
matplotlib.use("Agg")

class sdvel_on_map(object):
    """ A class to load and overlay various types of radar data """

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
                                    fill_alpha=1.0, fillContinents="None",
				    datetime=stime)

	# load the data, create sites and fovs for rads
	self.data = self._load_sddata()

        self.gridded_losvel = None

    def _load_sddata(self):
        """ Loads radar data for period of interest"""

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
	""" Displays the map"""
        import matplotlib.pyplot as plt
        plt.show()

        return

    def overlay_terminator(self, lat_range=[30., 90.], lon_range=[-180., 180.],
                           nlats=50, nlons=50, zorder=8):
        """Overlayes terminator in self.coords coordinate"""

        from davitpy.utils.calcSun import calcTerminator

        # calculate terminator 
        lats, lons, zen, term = calcTerminator(self.stime, lat_range,
                                               lon_range, nlats=nlats,
                                               nlons=nlons)
        # Plot the terminator line
        x, y = self.map_obj(term[:, 1], term[:, 0], coords="geo")
        self.map_obj.ax.scatter(x, y, facecolors='b',edgecolors='b', s=1.0)

        return

	
    def overlay_radName(self, fontSize=15, annotate=True):
	""" Overlay radar names """

	from davitpy import pydarn
	for i, r in enumerate(self.rads):
	    pydarn.plotting.overlayRadar(self.map_obj, codes=r, 
					 dateTime=self.stime,
					 fontSize=fontSize,
					 annotate=annotate)
	return

    def overlay_radFov(self, maxGate=70):
        """overlay radar fields of view"""

	from davitpy import pydarn

	for i,r in enumerate(self.rads):
	    pydarn.plotting.overlayFov(self.map_obj, codes=r, 
				       dateTime=self.stime,
				       maxGate=maxGate,
				       fovObj=self.fovs[i])
	return

    def overlay_raw_data(self, param="velocity",
			   gsct=0, fill=True,
			   velscl=1000., vel_lim=[-1000, 1000],
                           srange_lim=[450, 4000],
			   zorder=4,alpha=1,
			   cmap=None,norm=None):

   	"""Overlays raw LOS data from radars""" 

        from davitpy import pydarn
	import numpy as np
	import math
        from matplotlib.collections import PolyCollection, LineCollection
        
        losvel_mappable = None
        for i in range(len(self.data)):
            if self.data[i] is None:
                continue

            fov = self.fovs[i]
            site = self.sites[i]
	    myData = self.data[i]
            gs_flg,lines = [],[]
            if fill:
                verts,intensities = [],[]
            else:
                verts = [[],[]]
                intensities = []
        
            #loop through gates with scatter
            for myBeam in myData:
                for k in range(len(myBeam.fit.slist)):
                    if myBeam.fit.slist[k] not in fov.gates:
                        continue
                    if (myBeam.fit.slist[k] * myBeam.prm.rsep < srange_lim[0]) or\
                       (myBeam.fit.slist[k] * myBeam.prm.rsep > srange_lim[1]): 
			continue

                    if (myBeam.fit.v[k] < vel_lim[0]) or\
                       (myBeam.fit.v[k] > vel_lim[1]): 
			continue
                    
                    r = myBeam.fit.slist[k]
                    if fill:
                        x1,y1 = self.map_obj(fov.lonFull[myBeam.bmnum,r],
                                             fov.latFull[myBeam.bmnum,r])
                        x2,y2 = self.map_obj(fov.lonFull[myBeam.bmnum,r+1],
                                             fov.latFull[myBeam.bmnum,r+1])
                        x3,y3 = self.map_obj(fov.lonFull[myBeam.bmnum+1,r+1],
                                             fov.latFull[myBeam.bmnum+1,r+1])
                        x4,y4 = self.map_obj(fov.lonFull[myBeam.bmnum+1,r],
                                             fov.latFull[myBeam.bmnum+1,r])
        
                        # save the polygon vertices
                        verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))
        
                    else:
                        x1,y1 = self.map_obj(fov.lonCenter[myBeam.bmnum,r],
					     fov.latCenter[myBeam.bmnum,r])
                        verts[0].append(x1)
                        verts[1].append(y1)
                        x2,y2 = self.map_obj(fov.lonCenter[myBeam.bmnum,r+1],
					     fov.latCenter[myBeam.bmnum,r+1])
                        theta = math.atan2(y2-y1,x2-x1)
                        x2 = x1+myBeam.fit.v[k]*velscl*(-1.0)*math.cos(theta)
                        y2 = y1+myBeam.fit.v[k]*velscl*(-1.0)*math.sin(theta)
                        lines.append(((x1,y1),(x2,y2)))

                    if(gsct):
                        gs_flg.append(myBeam.fit.gflg[k])

		    #save the param to use as a color scale
		    if(param == 'velocity'):
			intensities.append(myBeam.fit.v[k])
        
            #do the actual overlay
            if fill :
                #if we have data
                if(verts != []):
                    if(gsct == 0):
                        inx = np.arange(len(verts))
                    else:
                        inx = np.where(np.array(gs_flg)==0)
                        x = PolyCollection(np.array(verts)[np.where(np.array(gs_flg)==1)],
					   facecolors='.3',linewidths=0,
					   zorder=zorder+1,alpha=alpha)
                        self.map_obj.ax.add_collection(x, autolim=True)
                    coll = PolyCollection(np.array(verts)[inx],
					   edgecolors='face',linewidths=0,
					   closed=False,zorder=zorder,
					   alpha=alpha, cmap=cmap,norm=norm)
                    #set color array to intensities
                    coll.set_array(np.array(intensities)[inx])
                    self.map_obj.ax.add_collection(coll, autolim=True)
            else:
                #if we have data
                if(verts != [[],[]]):
                    if(gsct == 0):
                        inx = np.arange(len(verts[0]))
                    else:
                        inx = np.where(np.array(gs_flg)==0)

                        #plot the ground scatter as open circles
                        x = self.map_obj.ax.scatter(\
				np.array(verts[0])[np.where(np.array(gs_flg)==1)],
                                np.array(verts[1])[np.where(np.array(gs_flg)==1)],
			        s=1.0, zorder=zorder,marker='o',linewidths=.5,
				facecolors='w',edgecolors='k')
                        self.map_obj.ax.add_collection(x, autolim=True)
        
                    # plot the i-s as filled circles
                    ccoll = self.map_obj.ax.scatter(np.array(verts[0])[inx],
						    np.array(verts[1])[inx],
						    s=1.0,zorder=zorder+1,marker='o',
						    c=np.array(intensities)[inx],
						    linewidths=.5, edgecolors='face',
						    cmap=cmap,norm=norm)
        
                    #set color array to intensities
                    self.map_obj.ax.add_collection(ccoll)

                    #plot the velocity vectors
                    coll = LineCollection(np.array(lines)[inx],linewidths=.5,
					   zorder=zorder+2,cmap=cmap,norm=norm)
                    coll.set_array(np.array(intensities)[inx])
                    self.map_obj.ax.add_collection(coll)
            if coll:
                losvel_mappable = coll
        self.losvel_mappable=losvel_mappable
	return
        
    def overlay_grids(self, lat_min=30, lat_max=90, dlat=1,
		      zorder=5, half_dlat_offset=True):
        """ Overlays grids cells on a map
    
        Parameters
        ----------
        half_dlat_offset : bool
            set to False implements NINT[360 sin(theta)] at theta = 89, 88, ... colatitude
            set to True implements NINT[360 sin(theta)] at theta = 89.5, 88.5, ... colatitude
        """
        import numpy as np
	import sys
	sys.path.append("../../data_preprocessing/")
	from bin_data import grids
        import datetime as dt
        from matplotlib.collections import PolyCollection

        # Create grids
	grds = grids(lat_min=lat_min, lat_max=lat_max,
		     dlat=dlat, half_dlat_offset=half_dlat_offset)
    
        # flatting all lats and lons 
        lons_all = np.array([])
        lons_all_E = np.array([])
        lons_all_W = np.array([])
        lats_all = np.array([])
        lats_all_N = np.array([])
        lats_all_S = np.array([])
        for i in range(len(grds.center_lats)):
            lons = [ item*grds.dlons[i] for item in np.arange(0.5, (grds.nlons[i]+0.5)) ]
            lons_E = [ item*grds.dlons[i] for item in np.arange(1, (grds.nlons[i]+1)) ]
            lons_W = [ item*grds.dlons[i] for item in np.arange(0, (grds.nlons[i])) ]
            lons_all = np.append(lons_all, lons)
            lons_all_E = np.append(lons_all_E, lons_E)
            lons_all_W = np.append(lons_all_W, lons_W)
            lats_all = np.append(lats_all, np.repeat(grds.center_lats[i], grds.nlons[i]))
            lats_all_N = np.append(lats_all_N,
				   np.repeat((grds.center_lats[i]+grds.dlat*0.5), grds.nlons[i]))
            lats_all_S = np.append(lats_all_S,
				   np.repeat((grds.center_lats[i]-grds.dlat*0.5), grds.nlons[i]))
        # plot grid
        x1,y1 = self.map_obj(lons_all_W,lats_all_S)
        x2,y2 = self.map_obj(lons_all_W,lats_all_N)
        x3,y3 = self.map_obj(lons_all_E,lats_all_N)
        x4,y4 = self.map_obj(lons_all_E,lats_all_S)
        verts = zip(zip(x1,y1), zip(x2,y2), zip(x3,y3), zip(x4,y4))
        coll = PolyCollection(np.array(verts), linewidth=0.07, 
			       facecolor='', edgecolor='k', zorder=zorder,
			       clip_on=True)
        self.map_obj.ax.add_collection(coll)

	return

    def calc_gridded_losvel(self, lat_min=30, lat_max=90, dlat=1,
			    srange_lim=[450, 4000],
			    min_npnts=1,
			    half_dlat_offset=False):
    
   	"""Calculates gridded LOS velocity 

        Parameters
        ----------
        half_dlat_offset : bool
            set to False implements NINT[360 sin(theta)] at
		 theta = 89, 88, ... colatitude
            set to True implements NINT[360 sin(theta)] at
		 theta = 89.5, 88.5, ... colatitude
        """
        import numpy as np
        import pandas as pd 
	import sys
	sys.path.append("../../data_preprocessing/")
	from bin_data import grids

        # Create grids
	grds = grids(lat_min=lat_min, lat_max=lat_max,
		     dlat=dlat, half_dlat_offset=half_dlat_offset)

        df_lst = []
        for i in range(len(self.data)):
            myData = self.data[i]
            if myData is None:
		df_lst.append(None)
                continue
            fov = self.fovs[i]
            site = self.sites[i]

            # gridded parameters
            gvels, glats, glons, gbmazms = [], [], [], []
            for myBeam in myData:
                if (len(myBeam.fit.slist)==0) or (myBeam is None):
                    continue
                for k in range(len(myBeam.fit.slist)):
                    if myBeam.fit.slist[k] not in fov.gates:
                        continue
                    if (myBeam.fit.slist[k] * myBeam.prm.rsep < srange_lim[0]) or\
                       (myBeam.fit.slist[k] * myBeam.prm.rsep > srange_lim[1]): 
			continue
#                    if (myBeam.fit.gflg[k]):     # filter out ground scatter
#                        continue
                    r = myBeam.fit.slist[k]
                    ilon = fov.lonCenter[myBeam.bmnum,r]
                    ilat = fov.latCenter[myBeam.bmnum,r]
                    ivel = myBeam.fit.v[k]
                    bmazm = myBeam.prm.bmazm

                    # grid the data
                    # grid latc
                    indx_lat = np.digitize([ilat], grds.lat_bins)
                    indx_lat = indx_lat[0]-1

                    # NOTE: the following way avoids nan in lat
		    try:
			glat = grds.center_lats[indx_lat]
		    except IndexError:
			glat = np.nan
                    glats.append(glat)

                    # grid lon
                    # NOTE: the following way avoids nan in lonc
                    try:
			indx_lon = np.digitize([ilon % 360], grds.lon_bins[indx_lat])
                        indx_lon = indx_lon[0]-1
			glon = grds.center_lons[indx_lat][indx_lon]
                    except IndexError:
			glon = np.nan
                    glons.append(glon)

                    gvels.append(ivel)
                    gbmazms.append(bmazm)
    
            #columns = ['latc', 'lonc', 'vel_'+self.rads[i], 'bmazm_'+self.rads[i]]
            #columns = ['latc', 'lonc', 'vel', 'bmazm']
            columns = ['latc', 'lonc', 'vel', 'bmazm', 'geolat_rad', 'geolon_rad']
            lat_rad= [site.geolat] * len(glats)
            lon_rad= [site.geolon] * len(glons)
            df_tmp = pd.DataFrame(zip(glats, glons, gvels, gbmazms, lat_rad, lon_rad), columns=columns)
            df_tmp = df_tmp.sort_values(['latc', 'lonc'])
            df_tmp = df_tmp.groupby(['latc', 'lonc'],
                                    as_index=False).\
                                    filter(lambda x: len(x)>= min_npnts)
            df_tmp = df_tmp.groupby(['latc', 'lonc'], as_index=False).median()
            df_lst.append(df_tmp)

	#self.gridded_losvel = pd.concat(df_lst)
	self.gridded_losvel = df_lst

        return 

    def overlay_gridded_losvel(self, vel_lim=[-1000, 1000], zorder=10,
			       cmap=None, norm=None, velscl=1000.):

        """ Overlays gridded LOS velocities on a map"""
    
        from matplotlib.collections import LineCollection
        verts = [[], []]
        intensities = []
        lines = []
        gridded_losvel_mappable=None
	for i in range(len(self.rads)): 
	    df = self.gridded_losvel[i]
	    try:
		#df = df.loc[(df['vel_'+self.rads[i]] >= vel_lim[0]) &\
		#	    (df['vel_'+self.rads[i]] <= vel_lim[1]), :]
		df = df.loc[(df['vel'] >= vel_lim[0]) &\
			    (df['vel'] <= vel_lim[1]), :]
	    except:
		continue
	    if df.shape[0] == 0:
		continue
	    site = self.sites[i]
	    lon_rad = site.geolon
	    lat_rad = site.geolat
	    xo,yo = self.map_obj(lon_rad, lat_rad, coords='geo')
	    xp,yp = self.map_obj(0, 90, coords='geo')
	    theta = np.arctan2(yp-yo,xp-xo)
	    #theta = theta - np.deg2rad(df['bmazm_'+self.rads[i]].tolist())
	    theta = theta - np.deg2rad(df['bmazm'].tolist())
	    x1,y1 = self.map_obj(df['lonc'].tolist(), df['latc'].tolist())
	    verts[0].extend(x1)
	    verts[1].extend(y1)
	    x1 = np.array(x1)
	    y1 = np.array(y1)
	    #x2 = x1+np.array(df['vel_'+self.rads[i]].tolist())*velscl*(-1.0)*np.cos(theta)
	    #y2 = y1+np.array(df['vel_'+self.rads[i]].tolist())*velscl*(-1.0)*np.sin(theta)
	    x2 = x1+np.array(df['vel'].tolist())*velscl*(-1.0)*np.cos(theta)
	    y2 = y1+np.array(df['vel'].tolist())*velscl*(-1.0)*np.sin(theta)
	    lines.extend(zip(zip(x1,y1),zip(x2,y2)))

	    #save the param to use as a color scale
	    #intensities.extend(df['vel_'+self.rads[i]].tolist())
	    intensities.extend(df['vel'].tolist())

	    #do the actual overlay
            #plot the i-s as filled circles
            ccoll = self.map_obj.ax.scatter(np.array(verts[0]),np.array(verts[1]),
					    s=3.0,zorder=zorder,marker='o',
					    c=np.array(intensities),
					    linewidths=.5, edgecolors='face',
					    cmap=cmap,norm=norm)
					    #linewidths=.5, edgecolors='face')
	    #self.map_obj.ax.add_collection(ccoll)

	    #plot the velocity vectors
	    coll = LineCollection(np.array(lines),linewidths=0.5,
				   zorder=zorder, cmap=cmap,norm=norm)
	    coll.set_array(np.array(intensities))
	    self.map_obj.ax.add_collection(coll)
            if coll:
                gridded_losvel_mappable=coll
        self.gridded_losvel_mappable=gridded_losvel_mappable

#        # resolved 2D velocity 
#        thetas = []   # the collection of angles of two pair radars' gridded LOS vel.
#        df = griddedVel(rads, scans, fovs)
#        if (df is not None):
#            dfg = df.groupby(['latc', 'lonc'], as_index=False).filter(lambda x: len(x) == 2)
#        if (df is not None) and (not dfg.empty):
#            for i in range(len(rads)):
#                site = sites[i]
#                lon_rad = site.geolon
#                lat_rad = site.geolat
#                xo,yo = self.map_obj(lon_rad, lat_rad, coords='geo')
#                xp,yp = self.map_obj(0, 90, coords='geo')
#                theta = np.arctan2(yp-yo,xp-xo)
#                #theta = theta - np.deg2rad(df['bmazm_'+rads[i]].tolist())
#                theta = theta - np.deg2rad(df['bmazm'].tolist())
#                thetas.append(theta)
#    
#            verts_r = [[], []]
#            #vel1 = df['vel_'+rads[0]].as_matrix()
#            #vel2 = df['vel_'+rads[1]].as_matrix()
#            vel1 = df['vel'].as_matrix()
#            vel2 = df['vel'].as_matrix()
#            vel1_theta = (np.sign(vel1)+1)/2 * np.pi + thetas[0]
#            vel2_theta = (np.sign(vel2)+1)/2 * np.pi + thetas[1]
#            dtheta = vel2_theta - vel1_theta   # the angle between two gridded LOS vel.
#            aa = np.cos(dtheta)
#            bb = np.sin(dtheta)
#            alpha1 = np.arctan((-aa + (np.abs(vel2)*1.0)/(np.abs(vel1))) / bb)
#            alpha1[np.where(alpha1 > np.pi/2)] += np.pi
#            #theta_r = theta[0] - alpha1
#            theta_r = vel1_theta + alpha1
#            #vels_r = np.true_divide(vel1, np.cos(alpha1))
#            vels_r = vel1 / np.cos(alpha1)
#            vels_r = np.abs(vels_r)
#    
#            xr1,yr1 = self.map_obj(df['lonc'].tolist(), df['latc'].tolist())
#            verts_r[0].extend(xr1)
#            verts_r[1].extend(yr1)
#            xr1 = np.array(xr1)
#            yr1 = np.array(yr1)
#    
#            xr2 = xr1+vels_r/velscl*(+1.0)*np.cos(theta_r)*dist
#            yr2 = yr1+vels_r/velscl*(+1.0)*np.sin(theta_r)*dist
#    
#            lines_r = zip(zip(xr1,yr1),zip(xr2,yr2))
#            #save the param to use as a color scale
#            intensities_r = vels_r.tolist()
#            ccoll_r = myFig.gca().scatter(np.array(verts_r[0]),np.array(verts_r[1]),
#                            s=3.0,zorder=11,marker='*', c=np.array(intensities_r),
#                            linewidths=.5, edgecolors='face',cmap=cmap,norm=norm)
#            #set color array to intensities
#            lcoll_r = LineCollection(np.array(lines_r),linewidths=1.0,zorder=12,cmap=cmap,norm=norm)
#            lcoll_r.set_array(np.array(intensities_r))
#            myFig.gca().add_collection(ccoll_r)
#            myFig.gca().add_collection(lcoll_r)
#    
#            return intensities_r,lcoll_r
#            #return intensities,lcoll
#        else:
#            return [None, None]
	return

    def overlay_2D_sdvel(self, npntslim_lfit=5,
                         rad_groups=[["wal", "bks"], ["fhe", "fhw"],
                                     ["cve", "cvw"], ["ade", "adw"]],
			 cmap=None,norm=None, velscl=1000.0, 
			 lfit_vel_max_lim=None,
                         lat_lim=[50., 90.], vel_err_ratio_lim=0.5,
			 all_lfitvel=False, hybrid_2Dvel=False,
			 nazmslim_pr_grid=1, OLS=False,
                         fitting_diagnostic_plot=False, fig_dir="./",
			 vel_scale=[-150, 150]):
        """Calculates the 2-D flow vectors and overlay them on a map"""
    
        from funcs import sdvel_lfit, plot_losvel_az
        import pandas as pd
        import numpy as np
        from matplotlib.collections import LineCollection

        lfitvel_mappable = None
        df_lfitvel_list = []
        # Do L-shell fitting for each radar groups
        for rds in rad_groups:
            try:
                indx = [self.rads.index(x) for x in rds]
                df_griddedvel = pd.concat([self.gridded_losvel[x] for x in indx], 
                                          keys=[self.rads[x] for x in indx])
            except ValueError:
                df_lfitvel_list.append(None)
                continue

            df_lfitvel = sdvel_lfit(self.map_obj, df_griddedvel,
                                    npntslim_lfit=npntslim_lfit, OLS=OLS)
            df_lfitvel_list.append(df_lfitvel)

            # Filter based on lat range
            if df_lfitvel is not None:
                df_lfitvel = df_lfitvel.loc[(df_lfitvel['latc'] >= lat_lim[0]) &\
                                            (df_lfitvel['latc'] <= lat_lim[1]), :]

            # Filter based on fitting quality
            if df_lfitvel is not None:
                df_lfitvel = df_lfitvel.loc[df_lfitvel['lfit_vel_err'].as_matrix() /\
                                            np.abs(df_lfitvel['lfit_vel'].as_matrix()) <\
                                            vel_err_ratio_lim, :]

            if all_lfitvel and (df_lfitvel is not None):

                # filter out the lfit velocity that exceedes lfit_vel_max_lim
                if lfit_vel_max_lim is not None:
                    df_lfitvel = df_lfitvel.where(df_lfitvel.lfit_vel <\
                                                  lfit_vel_max_lim).dropna()

                verts = [[], []]
                intensities = []
                lines = []
                xp, yp = self.map_obj(0, 90)    # North pole in the self.map_obj.coords system
                xo, yo = self.map_obj(df_lfitvel.lonc.tolist(),
                                      df_lfitvel.latc.tolist())
                the0 = np.arctan2(yp-np.array(yo),xp-np.array(xo))
                theta = the0 - np.deg2rad(df_lfitvel['lfit_azm'].tolist())
                x1,y1 = self.map_obj(df_lfitvel['lonc'].tolist(), df_lfitvel['latc'].tolist())
                verts[0].extend(x1)
                verts[1].extend(y1)
                x1 = np.array(x1)
                y1 = np.array(y1)

                x2 = x1+np.array(df_lfitvel['lfit_vel'].tolist())*velscl*(-1.0)*np.cos(theta)
                y2 = y1+np.array(df_lfitvel['lfit_vel'].tolist())*velscl*(-1.0)*np.sin(theta)

                lines.extend(zip(zip(x1,y1),zip(x2,y2)))
                #save the param to use as a color scale
                intensities.extend(df_lfitvel['lfit_vel'].tolist())

                #do the actual overlay
                #plot the i-s as filled circles
                ccoll = self.ax.scatter(np.array(verts[0]),np.array(verts[1]),
                                #s=.1*np.array(intensities[1])[inx],zorder=10,marker='o',
                                s=2.0,zorder=10,marker='o', c=np.abs(np.array(intensities)),
                                linewidths=.5, edgecolors='face',cmap=cmap,norm=norm)

                self.ax.add_collection(ccoll)
                #plot the velocity vectors
                lcoll = LineCollection(np.array(lines),linewidths=0.5,zorder=12,cmap=cmap,norm=norm)
                lcoll.set_array(np.abs(np.array(intensities)))
                self.ax.add_collection(lcoll)
            else:
                lcoll = None

            # Seperate overlaped gridded los vel from non-overlaped ones
            df1_griddedvel = df_griddedvel.groupby(['latc', 'lonc'], as_index=False).\
                                           filter(lambda x: len(x)==1)
            df2_griddedvel = df_griddedvel.groupby(['latc', 'lonc'], as_index=False).\
                                           filter(lambda x: len(x)==2)
            #df1_lfitvel = df_lfitvel.groupby(['latc', 'lonc']).\
            #                         filter(lambda x: len(x)==1)
            if hybrid_2Dvel:
                pass
#                # Find 2D vector if overlapped los vels exist
#                # if not then use the L-shell fitted 2D vels.
#                verts = [[], []]
#                intensities = []
#                lines = []
#                if nazmslim_pr_grid < 2:
#                    if (not df1_lfitvel.empty):
#                        xp, yp = self.map_obj(0, 90)    # geomagnetic pole position on the map
#                        xo, yo = self.map_obj(df1_lfitvel.lonc.tolist(), df1_lfitvel.latc.tolist())
#                        the0 = np.arctan2(yp-np.array(yo),xp-np.array(xo))
#                        theta = the0 - np.deg2rad(df1_lfitvel['lfit_azm'].tolist())
#                        x1,y1 = self.map_obj(df1_lfitvel['lonc'].tolist(), df1_lfitvel['latc'].tolist())
#                        verts[0].extend(x1)
#                        verts[1].extend(y1)
#                        x1 = np.array(x1)
#                        y1 = np.array(y1)
#                        x2 = x1+np.array(df1_lfitvel['lfit_vel'].tolist())/velscl*(-1.0)*np.cos(theta)*dist
#                        y2 = y1+np.array(df1_lfitvel['lfit_vel'].tolist())/velscl*(-1.0)*np.sin(theta)*dist
#
#                        lines.extend(zip(zip(x1,y1),zip(x2,y2)))
#                        #save the param to use as a color scale
#                        intensities.extend(df1_lfitvel['lfit_vel'].tolist())
#
#                # for df2_merged
#                df2_merged = merge_2losvecs(self.map_obj, df2_griddedvel)
#                if df2_merged is not None:
#                    xr1,yr1 = self.map_obj(df2_merged['lonc'].tolist(), df2_merged['latc'].tolist())
#                    verts[0].extend(xr1)
#                    verts[1].extend(yr1)
#                    xr1 = np.array(xr1)
#                    yr1 = np.array(yr1)
#                    vel_r = df2_merged['vel_2d'].as_matrix()
#                    theta_r = np.deg2rad(df2_merged['theta_2d'].as_matrix())
#                    xr2 = xr1+vel_r/velscl*(+1.0)*np.cos(theta_r)*dist
#                    yr2 = yr1+vel_r/velscl*(+1.0)*np.sin(theta_r)*dist
#
#                    lines.extend(zip(zip(xr1,yr1),zip(xr2,yr2)))
#                    #save the param to use as a color scale intensities.extend(vel_r.tolist())
#                    intensities.extend(df2_merged['vel_2d'].tolist())
#
#                # for dfge3_lfitvel
#                dfge3_griddedvel = df_griddedvel.groupby(['latc', 'lonc'], as_index=False).filter(lambda x: len(x)>=3)
#                dfgge3 = dfge3_griddedvel.groupby(['latc', 'lonc'], as_index=False)
#                groups_lfit=[]
#                for indx, group in dfgge3:
#                    group_lfit = sdvel_lfit(self.map_obj, group, npntslim_lfit=3)
#                    if group_lfit is not None:
#                        groups_lfit.append(group_lfit)
#                if (not groups_lfit==[]):
#                    dfge3_lfitvel = pd.concat(groups_lfit)
#                    xp, yp = self.map_obj(0, 90)    # geomagnetic pole position on the map
#                    xo, yo = self.map_obj(dfge3_lfitvel.lonc.tolist(), dfge3_lfitvel.latc.tolist())
#                    the0 = np.arctan2(yp-np.array(yo),xp-np.array(xo))
#                    theta = the0 - np.deg2rad(dfge3_lfitvel['lfit_azm'].tolist())
#                    x1,y1 = self.map_obj(dfge3_lfitvel['lonc'].tolist(), dfge3_lfitvel['latc'].tolist())
#                    verts[0].extend(x1)
#                    verts[1].extend(y1)
#                    x1 = np.array(x1)
#                    y1 = np.array(y1)
#                    x2 = x1+np.array(dfge3_lfitvel['lfit_vel'].tolist())/velscl*(-1.0)*np.cos(theta)*dist
#                    y2 = y1+np.array(dfge3_lfitvel['lfit_vel'].tolist())/velscl*(-1.0)*np.sin(theta)*dist
#
#                    lines.extend(zip(zip(x1,y1),zip(x2,y2)))
#                    #save the param to use as a color scale
#                    intensities.extend(dfge3_lfitvel['lfit_vel'].tolist())
#
#
#                #do the actual overlay
#                #plot the i-s as filled circles
#                ccoll = self.ax.scatter(np.array(verts[0]),np.array(verts[1]),
#                                #s=.1*np.array(intensities[1])[inx],zorder=10,marker='o',
#                                s=3.0,zorder=10,marker='o', c=np.abs(np.array(intensities)),
#                                linewidths=.5, edgecolors='face',cmap=cmap,norm=norm)
#
#                self.ax.add_collection(ccoll)
#                #plot the velocity vectors
#                lcoll = LineCollection(np.array(lines),linewidths=0.5,zorder=12,cmap=cmap,norm=norm)
#                lcoll.set_array(np.abs(np.array(intensities)))
#                self.ax.add_collection(lcoll)

            if lcoll:
                lfitvel_mappable = lcoll
            self.lfitvel_mappable=lfitvel_mappable

            if fitting_diagnostic_plot:
                if df_lfitvel is not None:
                    color_list = ['r', 'b', 'g', 'c', 'm', 'k']
                    latc_list = [x + 0.5 for x in range(53, 63)]
                    #latc_list=None
                    rds_remain = [x for x in df_griddedvel.index.get_level_values(0).unique()]
                    plot_losvel_az(rds_remain, df_lfitvel, color_list,
                                   self.stime, self.interval,
                                   latc_list=latc_list, fig_dir=fig_dir,
				   vel_scale=vel_scale)
        self.lfitvel = df_lfitvel_list
        return

    def overlay_tec(self, ctime=None, cmap=None, norm=None,
		    zorder=5,
		    inpDir = "/sd-data/med_filt_tec/"):

        """Overlays the GPS TEC data
        """

        import pandas as pd
        from funcs import convert_to_datetime
    
        # Read the median filtered TEC data
        inpColList = [ "dateStr", "timeStr", "Mlat",\
                              "Mlon", "med_tec", "dlat", "dlon" ]
        if ctime is None:
            ctime = self.stime
        inpFile = inpDir + "tec-medFilt-" + ctime.strftime("%Y%m%d") + ".txt"
        medFiltTECDF = pd.read_csv(inpFile, delim_whitespace=True,
                                   header=None, names=inpColList)
        medFiltTECDF["datetime"] = medFiltTECDF.apply(convert_to_datetime, axis=1)
    
        # Find the time closed to time interval of interest
        stm = ctime - dt.timedelta(minutes=5)
        etm = ctime + dt.timedelta(minutes=5)
        df = medFiltTECDF.loc[(medFiltTECDF.datetime >= stm) &\
                              (medFiltTECDF.datetime < etm), :]

	# Overlay the tec data on to a map
	x1, y1 = self.map_obj(df.Mlon.as_matrix(),
			      df.Mlat.as_matrix(), coords="mag")

	ccoll = self.map_obj.ax.scatter(x1, y1, s=30.0, zorder=zorder,
					marker="s", c=df.med_tec.as_matrix(),
					linewidths=.5, edgecolors='face',
					cmap=cmap,norm=norm)

#	
#	ccoll = self.map_obj.pcolor(x1, y1, df.med_tec.as_matrix(),
#				    alpha=1.0, zorder=zorder,
#				    vmin=0., vmax=5.,
#				    cmap=cmap, norm=norm)

        self.tec_mappable = ccoll

        return 

    def overlay_poes(self, pltDate=None, selTime=None,
		     satList=["m01", "n15", "n19", "n18"],
		     plotCBar=False, cbar_shrink=0.8, 
		     rawSatDir="../../data/poes/raw/",
		     inpFileDir="../../data/poes/bnd/"):
        """Overlays POES Satellite data and
        the estimate Plassmapause Boundary

        NOTE: POES data has to be downloaded and processed. 
        
        """
        #import sys
        #sys.path.insert(0, "/home/muhammad/softwares/sataurlib/")
        from poes import poes_plot_utils
	import datetime as dt
        import os

        if selTime is None:
            selTime = self.stime
	if pltDate is None:
	    pltDate = dt.datetime(selTime.year, selTime.month, selTime.day)
	inpFileName = inpFileDir + "poes-fit-" + pltDate.strftime("%Y%m%d") + ".txt"
        if os.path.isfile(inpFileName):
            poesPltObj = poes_plot_utils.PlotUtils(pltDate, pltCoords=self.coords)
            poesPltObj.overlay_closest_sat_pass(selTime, self.map_obj, self.ax,
                                                rawSatDir, satList=satList,
                                                plotCBar=plotCBar,
                                                timeFontSize=4., timeMarkerSize=2.,
                                                overlayTimeInterval=1, timeTextColor="red",
                                                cbar_shrink=cbar_shrink)
            # two ways to overlay estimated boundary!
            # poesPltObj.overlay_equ_bnd(selTime, self.map_obj, self.ax,rawSatDir)
            poesPltObj.overlay_equ_bnd(selTime, self.map_obj, self.ax,\
                                       inpFileName=inpFileName,
                                       linewidth=1, linecolor="red", line_zorder=7)

        return

    def overlay_ssusi(self, pltDate=None, inpTime=None,
                      plotType="d135", timeDelta=40.,
		      satList=["F18"],
		      plotCBar=False, cbar_shrink=0.8, 
                      ssusiCmap="Greens", alpha=0.6,
		      inpDir="../../data/ssusi/prcsd/"):

        """Overlays DMSP Satellite SSUSI data

        NOTE: SSUSI data has to be downloaded and processed. 
        
        """
        #import sys
        #sys.path.insert(0, "/home/muhammad/softwares/sataurlib/")
        from imagers.ssusi import ssusi_utils
	import datetime as dt
        import os

        if inpTime is None:
            inpTime = self.stime
	if pltDate is None:
	    pltDate = dt.datetime(inpTime.year, inpTime.month, inpTime.day)
        for sat in satList:
            inpFileName = inpDir + sat + "/" + pltDate.strftime("%Y%m%d") + ".txt"
            if os.path.isfile(inpFileName):
                ssusiPltObj = ssusi_utils.UtilsSsusi(inpDir, pltDate)
                fDict = ssusiPltObj.filter_data_by_time(inpTime, timeDelta=timeDelta)
                ssusiPltObj.overlay_sat_data(fDict, self.map_obj, self.ax,
                                             satList=satList, inpTime=inpTime, 
                                             plotType=plotType, coords=self.coords,
                                             alpha=alpha, plotCBar=plotCBar,
                                             overlayTimeInterval=1, timeColor="red",
                                             timeFontSize=4.,
                                             ssusiCmap=ssusiCmap,
                                             cbar_shrink=cbar_shrink)

        return

def add_cbar(fig, mappable, label="Velocity [m/s]", cax=None,
             ax=None, shrink=0.65, title_size=14,
             ytick_label_size=12):

    # add color bar
    cbar=fig.colorbar(mappable, ax=ax, cax=cax, shrink=shrink,
                      orientation="vertical", drawedges=False)

#    #define the colorbar labels
#    l = []
#    for i in range(0,len(bounds)):
#        if i == 0 or i == len(bounds)-1:
#            l.append(' ')
#            continue
#        l.append(str(int(bounds[i])))
#    cbar.ax.set_yticklabels(l)
    cbar.ax.tick_params(axis='y',direction='out')
    cbar.set_label(label)

    #set colorbar ticklabel size
    for ti in cbar.ax.get_yticklabels():
        ti.set_fontsize(ytick_label_size)
    cbar.set_label(label, size=title_size)
    cbar.extend='max'

if __name__ == "__main__":
    import datetime as dt
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap as lcm
    from matplotlib.colors import BoundaryNorm, Normalize
    import matplotlib.cm
    import os

    # Control parameters 
    sddata_type = "raw_los" 
    #sddata_type = "grid_los" 
    #sddata_type = "lfitvel" 
    overlay_poes_data = False
    overlay_ssusi_data = True
    overlay_tec_data = True
    overlay_terminator_line = False
    if overlay_ssusi_data:
        fig_txt = sddata_type + "_ssusi"
        rawlos_srange_lim=[450, 4000]
    elif overlay_poes_data:
        fig_txt = sddata_type + "_poes"
        rawlos_srange_lim=[450, 4000]
    else:
        fig_txt = sddata_type + ""
        rawlos_srange_lim=[450, 4000]

    vel_scale=[-100,100]
    vel_scale_los_az=[-200,200]

#    vel_scale=[-50,60]
#    vel_scale_los_az=[-60,60]

    #stime = dt.datetime(2011, 5, 16, 8, 0)
    #stime = dt.datetime(2011, 5, 29, 3, 0)
    #stime = dt.datetime(2011, 7, 5, 4, 0)
    #stime = dt.datetime(2013, 1, 2, 7, 0)
    #stime = dt.datetime(2013, 1, 18, 3, 40)
    #stime = dt.datetime(2013, 11, 14, 5, 30)
    #stime = dt.datetime(2013, 2, 4, 5, 30)
    #stime = dt.datetime(2013, 2, 21, 3, 30)
    #stime = dt.datetime(2013, 2, 24, 4, 0)
    #stime = dt.datetime(2013, 3, 17, 5, 30)
    #stime = dt.datetime(2013, 3, 19, 4, 30)
    #stime = dt.datetime(2013, 4, 15, 8, 00)
    #stime = dt.datetime(2013, 5, 14, 7, 20)
    #stime = dt.datetime(2013, 5, 19, 5, 0)
    #stime = dt.datetime(2013, 11, 8, 4, 0)
    #stime = dt.datetime(2013, 11, 16, 9, 0)
    #stime = dt.datetime(2013, 11, 17, 5, 50)
    #stime = dt.datetime(2013, 12, 27, 5, 50)
    #stime = dt.datetime(2014, 7, 12, 9, 0)
    stime = dt.datetime(2014, 12, 16, 13, 30)
    #stime = dt.datetime(2015, 4, 9, 5, 0)
    if overlay_poes_data or overlay_ssusi_data : 
        interval = 30*60    # half an hour
        nums_itr = 1     # number of interation
    else:
        interval = 2*60
        nums_itr = 1*30     # number of interation
        #nums_itr = 1     # number of interation

    dtms = [stime + dt.timedelta(seconds=x) for x in range(0, interval * nums_itr, interval)]
    for stime in dtms:

	# Create a folder for a date
	fig_dir = "../plots/scan_plot/" + stime.strftime("%Y%m%d") + "/"
	if not os.path.exists(fig_dir):
	    os.makedirs(fig_dir)

	etime = stime+dt.timedelta(seconds=interval)
	coords = "mlt"

	#rads = ["cve", "cvw"]
	#rads = ["wal", "bks", "fhe", "fhw", "cve", "cvw"]
        #channel=None
        # NOTE: Do not forget to set the channel
	rads = ["wal", "bks", "fhe", "fhw", "cve", "cvw", "ade", "adw"]
        channel = [None, None, None, None, None, None, 'all', 'all']


	fig, ax = plt.subplots(figsize=(8,6))
    #    # customized cmap
    #    cmj = matplotlib.cm.jet
    #    cmpr = matplotlib.cm.prism
    #    cmap = lcm([cmj(.95), cmj(.85), cmj(.79), cmpr(.142), cmj(.45), cmj(.3), cmj(.1)])
    #    bounds = np.round(np.linspace(vel_scale[0], vel_scale[1], 7))
    #
	color_list = ['purple', 'b', 'c', 'g', 'y', 'r']
	cmap_lfit = lcm(color_list)
	#bounds_lfit = [0., 8, 17, 25, 33, 42, 10000]
	bounds_lfit = [int(x) for x in np.linspace(0, vel_scale[1], 6)]
	bounds_lfit.append(1000)
	norm_lfit = BoundaryNorm(boundaries=bounds_lfit,
				 ncolors=len(bounds_lfit)-1)

	#cmap_lfit = "nipy_spectral"
	#norm_lfit = Normalize(vmin=0,vmax=vel_scale[1])

	cmap = "jet_r"
	#norm = None
	norm = Normalize(vmin=vel_scale[0],vmax=vel_scale[1])

	# create an obj
        if overlay_poes_data or overlay_ssusi_data:
            map_lat0 = 90
            map_lon0= 0
            map_width=80*111e3 
            map_height=80*111e3 
            #map_lat0 = 67
            #map_lon0=0
            #map_width=80*111e3 
            #map_height=45*111e3 
        else:
            #map_lat0 = 67
            #map_lon0=0
            #map_width=80*111e3 
            #map_height=45*111e3 
            map_lat0 = 90
            map_lon0 = 0
            map_width = 80*111e3 
            map_height = 80*111e3 

	obj = sdvel_on_map(ax, rads, stime, interval=interval,
			   map_lat0=map_lat0, map_lon0=0,
			   map_width=map_width, 
			   map_height=map_height, 
			   #map_lat0=90, map_lon0=0,
			   #map_width=60*111e3, 
			   #map_height=60*111e3, 
			   map_resolution='l', 
			   coords=coords,
			   channel=channel,
			   fileType="fitacf")

#####################################################################
        if sddata_type == "raw_los":
            # Overlay LOS velocity data 
            obj.overlay_raw_data(param="velocity",
                                 gsct=0, fill=True,
                                 velscl=4000., vel_lim=[-1000, 1000],
                                 srange_lim=rawlos_srange_lim,
                                 zorder=4,alpha=0.7,
                                 cmap=cmap,norm=norm)

            # Add colorbar for LOS Vel.
            add_cbar(fig, obj.losvel_mappable, label="Velocity [m/s]", cax=None,
                     ax=None, shrink=0.5, title_size=14, ytick_label_size=10)
#####################################################################
	# Overlay Radar Names
	obj.overlay_radName()

#        # Overlay Radar FoVs
#        obj.overlay_radFov(maxGate=50)

	# Overlay Grids
	obj.overlay_grids(lat_min=20, lat_max=90, dlat=1,
			  zorder=2, half_dlat_offset=False)

#####################################################################
        if sddata_type == "lfitvel" or sddata_type == "grid_los":
            # Calculate gridded LOS velocity
            obj.calc_gridded_losvel(lat_min=30, lat_max=90, dlat=1,
                                    srange_lim=[450, 2000],
                                    min_npnts=1,
                                    half_dlat_offset=False)

        if sddata_type == "grid_los":
            # Overlay gridded LOS velocity
            obj.overlay_gridded_losvel(zorder=10, vel_lim=[-200, 200],
        			       cmap=cmap, norm=norm, velscl=4000.)

            # Add colorbar for gridded LOS Vel.
            add_cbar(fig, obj.gridded_losvel_mappable, label="Velocity [m/s]", cax=None,
                     ax=None, shrink=0.5, title_size=14, ytick_label_size=10)


        if sddata_type == "lfitvel":
            # Overlay L-shell fitted velocity
            obj.overlay_2D_sdvel(npntslim_lfit=10, lat_lim=[53, 70],
                                 rad_groups=[["wal", "bks"], ["fhe", "fhw"],
                                             ["cve", "cvw"], ["ade", "adw"]],
                                 cmap=cmap_lfit,norm=norm_lfit, velscl=4000.0, 
                                 lfit_vel_max_lim=None, vel_err_ratio_lim=0.2,
                                 all_lfitvel=True, hybrid_2Dvel=False,
                                 nazmslim_pr_grid=1, OLS=False,
                                 fitting_diagnostic_plot=True, fig_dir=fig_dir, 
                                 vel_scale=vel_scale_los_az)

            # Add colorbar for L-Shell Fit Vel.
            if obj.lfitvel_mappable:
                add_cbar(fig, obj.lfitvel_mappable, label="Velocity [m/s]", cax=None,
                         ax=None, shrink=0.5, title_size=14, ytick_label_size=10)

#####################################################################
        if overlay_tec_data:
            # Overlay GPS TEC data
            cmap='gist_gray_r'
            #cmap='jet'
            norm = Normalize(vmin=0., vmax=10.)
            obj.overlay_tec(ctime=None, cmap=cmap, norm=norm,
                            zorder=1, inpDir = "/sd-data/med_filt_tec/")

            # Add colorbar for TEC.
            add_cbar(fig, obj.tec_mappable, label="TEC [TECU]", cax=None,
                     ax=None, shrink=0.5, title_size=14, ytick_label_size=10)
#####################################################################
        if overlay_poes_data:
            # Overlay POES data
            obj.overlay_poes(pltDate=None, selTime=None,
                             #satList=["m01", "m02", "n15", "n16", "n17", "n18", "n19"],
                             satList=["m01", "n15", "n16", "n17", "n18", "n19"],
                             plotCBar=False, cbar_shrink=0.5,
                             rawSatDir="../../data/poes/raw/",
                             inpFileDir="../../data/poes/bnd/")

#####################################################################
        if overlay_ssusi_data:
            obj.overlay_ssusi(pltDate=None, inpTime=None,
                              plotType="d135", timeDelta=40.,
                              satList=["F16", "F17", "F18", "F19"],
                              plotCBar=False, cbar_shrink=0.8, 
                              ssusiCmap="Greens", alpha=0.6,
                              inpDir="../../data/ssusi/prcsd/")


#####################################################################
        if overlay_terminator_line:
            obj.overlay_terminator(lat_range=[30., 90.], lon_range=[-180., 180.],
                                   nlats=50, nlons=50, zorder=8)

#####################################################################

	# Add title 
	ax.set_title(stime.strftime('%b/%d/%Y   ') +\
		     stime.strftime('%H:%M - ')+\
		     etime.strftime('%H:%M  UT'))

	# Save the figure
        #fig_name = "test"
	fig_name = fig_txt + "_" +\
		   stime.strftime("%Y%m%d.%H%M") + "_to_" +\
		   etime.strftime("%Y%m%d.%H%M")
	fig.savefig( fig_dir + fig_name +\
		    ".png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    #    obj.show_map()

