class grids(object):

    def __init__(self, lat_min=50, lat_max=90, dlat=1, half_dlat_offset=False):
        """ This class is used to create lat-lon-azm bins on a map.

        Parameters
        ----------
        lat_min : int or float 
            The minimum latitude where gridding stops.
        lat_max : int or float 
            The maximum latitude where gridding stops.
        dlat : int or float 
            Width of each grid-cell
        half_dlat_offset : bool, default to False
            Note: half_dlat_offset=False implements NINT[360 sin(theta)] 
                  (determines longitudinal width) at theta = 89, 88, ... colatitude.
                  half_dlat_offset=True implements NINT[360 sin(theta)]
                  (determines longitudinal width) at theta = 89.5, 88.5, ... colatitude.

        Returns
        -------
        Nothing

        """

        import numpy as np

        self.lat_min = lat_min
        self.lat_max = lat_max
        self.dlat = dlat
        self.half_dlat_offset = half_dlat_offset
        self.center_lats = [x + 0.5*dlat for x in np.arange(lat_min, lat_max, dlat)] 
        if half_dlat_offset:
            self.nlons = [round(360. * np.sin(np.deg2rad(90.-lat))) for lat in self.center_lats]
        else:
            # In this case, grid-cells next to the polar point share the same point, the polar point.
            self.nlons = [round(360. * np.sin(np.deg2rad(90.-(lat-0.5*dlat)))) for lat in self.center_lats]
        self.dlons = [360./nn for nn in self.nlons]

        # lat and lon bins (the edges of each grid-cell)
        self.lat_bins = [x for x in np.arange(lat_min,lat_max+dlat,dlat)] 
        self.lon_bins, self.center_lons = self._create_lonbins()
        
        # azimuthal bins and their centers
        # Note: zero azm indicate the direction towards the mag north
        self.azm_bins = [x for x in range(0, 370, 10)]
        self.center_azms = [x for x in range(5, 365, 10)]

        return

    def _create_lonbins(self):
        """ creates longitudinal bins """

        import numpy as np

        lon_bins = []
        center_lons = []      # a list of lists of lons
        for i in range(len(self.center_lats)):
            lon_tmp = [ round(item*self.dlons[i],2) for item in np.arange(0.5, self.nlons[i]+0.5) ]
            center_lons.append(lon_tmp)
            lon_tmp = [ item*self.dlons[i] for item in np.arange(self.nlons[i]) ]
            lon_tmp.append(360.) 
            lon_bins.append(lon_tmp)

        return lon_bins, center_lons 
        

def bin_to_grid(rad, bmnum, stm=None, etm=None, ftype="fitacf",
		coords = "mlt",
                config_filename="../mysql_dbconfig_files/config.ini",
                section="midlat", db_name=None):

    """ bins the data into mlat-mlt-azm.

    Parameters
    ----------
    rad : str
        Three-letter radar code
    bmnum : int
        Radar beam
    stm : datetime.datetime
        The start time.
        Default to None, in which case takes the earliest in db.
    etm : datetime.datetime
        The end time.
        Default to None, in which case takes the latest time in db.
        NOTE: if stm is None then etm should also be None, and vice versa.
    ftype : str
        SuperDARN file type
    coords : str
        Coordinates in which the binning process takes place.
        Default to "mlt. Can be "geo" as well. 
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    db_name : str, default to None
        Name of the MySQL db to which iscat data has been written

    Note
    ----
        0 xxx_gazmc directs towards magnetic (or geo) north. 180 gazmc directs towards south.
        gazmc (gridded azimuthal center) spans from 5 - 355 degrees. 
    
    """

    import numpy as np
    import datetime as dt
    from mysql.connector import MySQLConnection
    import sys
    sys.path.append("../")
    from mysql_dbutils.db_config import db_config
    import logging


    # create grid points
    grds = grids(lat_min=35, lat_max=90, dlat=1, half_dlat_offset=False)

    # read the db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make db connection
    if db_name is None:
        db_name = rad + "_iscat_" + ftype
    try:
        conn = MySQLConnection(database=db_name, **config_info)
    except Exception, e:
        logging.error(e, exc_info=True)
    cur = conn.cursor(buffered=True)

    # check whether table_name exists. If not, do nothing
    table_name = rad + "_bm" + str(bmnum)
    command = "SHOW TABLES LIKE '{tb}'".format(tb=table_name)
    cur.execute(command)
    if not cur.fetchall():
        return

    # add new columns
    if coords == "mlt":
        col_glatc = "mag_glatc"   # glatc -> gridded latitude center
        col_gltc = "mag_gltc"   # mlt hour in degrees
        col_gazmc = "mag_gazmc"   # gazmc -> gridded azimuthal center
    if coords == "geo":
        col_glatc = "geo_glatc"
        col_gltc = "geo_gltc"    # local time in degrees
        col_gazmc = "geo_gazmc"
    try:
        command ="ALTER TABLE {tb} ADD COLUMN {glatc} TEXT".format(tb=table_name, glatc=col_glatc) 
        conn.cursor.execute(command)
    except:
        # pass if the column glatc exists
        pass
    try:
        command ="ALTER TABLE {tb} ADD COLUMN {glonc} TEXT".format(tb=table_name, glonc=col_gltc) 
        conn.cursor.execute(command)
    except:
        # pass if the column gltc exists
        pass
    try:
        command ="ALTER TABLE {tb} ADD COLUMN {gazmc} TEXT".format(tb=table_name, gazmc=col_gazmc) 
        conn.cursor.execute(command)
    except:
        # pass if the column gazmc exists
        pass



    # do the convertion to all the data in db if stm and etm are all None
    if coords == "mlt":
	col_latc = "mag_latc"
	col_ltc = "mag_ltc"
	col_azmc = "mag_azmc"
    if coords == "geo":
	col_latc = "geo_latc"
	col_ltc = "geo_ltc"
	col_azmc = "geo_azmc"

    if (stm is not None) and (etm is not None):
        command = "SELECT {latc}, {lonc}, {azm}, datetime FROM {tb} " +\
                  "WHERE datetime BETWEEN '{sdtm}' AND '{edtm}' ORDER BY datetime"
        command.format(tb=table_name, sdtm=stm, edtm=etm,
			latc=col_latc, lonc=col_ltc, azm=col_azmc)

    # do the convertion to the data between stm and etm if any of them is None
    else:
        command = "SELECT {latc}, {lonc}, {azm}, datetime FROM {tb} ORDER BY datetime".\
		  format(tb=table_name, latc=col_latc, lonc=col_ltc, azm=col_azmc)

    try:
        cur.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)
    rows = conn.cursor.fetchall() 

    # do the conversion row by row
    if rows != []:
        for row in rows:
            rowid, latc, lonc, azm, date_time= row
            if latc:

                # convert string to a list of float
                latc = [float(x) for x in latc.split(",")]
                lonc = [float(x) for x in lonc.split(",")]
                azm = [float(x) for x in azm.split(",")]

                # grid the data
                # grid latc
                indx_latc = np.digitize(latc, grds.lat_bins)
                indx_latc = [x-1 for x in indx_latc]
                glatc = [grds.center_lats[x] for x in indx_latc]

                # grid lonc
                indx_lonc = [np.digitize(lonc[i], grds.lon_bins[indx_latc[i]]) 
                             for i in range(len(lonc))]
                indx_lonc = [x-1 for x in indx_lonc]
                glonc = [grds.center_lons[indx_latc[i]][indx_lonc[i]]\
                         for i in range(len(lonc))]

                # grid azm
                indx_azmc = np.digitize(azm, grds.azm_bins)
                indx_azmc = [x-1 for x in indx_azmc]
                gazmc = [grds.center_azms[x] for x in indx_azmc]

                # convert to comma seperated text
                glatc =",".join([str(x) for x in glatc])
                glonc =",".join([str(x) for x in glonc])
                gazmc =",".join([str(x) for x in gazmc])

                # update the table
		if coords == "mlt":
		    command = "UPDATE {tb} SET mag_glatc='{glatc}', " +\
			      "mag_gltc='{glonc}', mag_gazmc='{gazmc}' " +\
			      "WHERE datetime = '{dtm}'"
		if coords == "geo":
		    command = "UPDATE {tb} SET geo_glatc='{glatc}', " +\
			      "geo_gltc='{glonc}', geo_gazmc='{gazmc}' " +\
			      "WHERE datetime = '{dtm}'"
		command.format(tb=table_name, glatc=col_glatc,
			       glonc=col_gltc, gazmc=col_gazmc, dtm=date_time)

                # check db connection before updating
                if not conn.is_connected():
                    conn.reconnect()
		# update
                cur.execute(command)

        
        # check db connection
        if not conn.is_connected():
            conn.reconnect()

        # commit the results
        try:
            conn.commit()
        except Exception, e:
            logging.error(e, exc_info=True)

    # close the db connection
    conn.close()
    return

def worker(rad, bmnum, stm=None, etm=None, ftype="fitacf", coords="mlt",
	   config_filename="../mysql_dbconfig_files/config.ini",
           section="midlat", db_name=None):
    """ A worker function to be used for parallel computing """

    import datetime as dt

    if db_name is None:
        db_name = rad + "_iscat_" + ftype

    # start running geo_to_mlt
    t1 = dt.datetime.now()
    if coords=="geo":
        print("start binning in geo coords. for beam " + str(bmnum) + " of " +\
              rad + " for period between " + str(stm) + " and " + str(etm))
    elif coords=="mlt":
        print("start binnng in MLAT-MLT coords. for beam " + str(bmnum) + " of " +\
              rad + " for period between " + str(stm) + " and " + str(etm))
    bin_to_grid(rad, bmnum, stm=stm, etm=etm, ftype=ftype,
		coords=coords, config_filename=config_filename,
                section=section, db_name=db_name)
    print("Binned values have been written to db for beam " + str(bmnum) +\
           " of " + rad + " for period between " + str(stm) + " and " + str(etm))

    t2 = dt.datetime.now()
    print("Finishing binning for beam " + str(bmnum) +\
           " of " + rad + " for period between " + str(stm) + " and " +\
           str(etm) + " took " + str((t2-t1).total_seconds() / 60.) + " mins\n")

    return

def main(run_in_parallel=True):
    """ Call the functions above. Acts as an example code.
    Multiprocessing has been implemented to do parallel computing.
    A unit process is for a radar beam (i.e. a db table)"""

    import datetime as dt
    import multiprocessing as mp
    import logging
    
    # create a log file to which any error occured between client and
    # MySQL server communication will be written.
    logging.basicConfig(filename="./log_files/geo_to_mlt_hok.log",
                        level=logging.INFO)
    
    # input parameters
    stm = None 
    etm = None 
    ftype = "fitacf"
    coords="mlt"         # set this to "geo" if you want to remain in "geo" coords
    db_name = None       # if set to None default iscat db would be read. 
    
    # run the code for the following radars in parallel
    rad_list = ["hok"]
    #rad_list = ["ade", "adw"]
    #rad_list = ["tig", "unw"]
    #rad_list = ["bpk"]
    #rad_list = ["bks", "wal", "fhe", "fhw", "cve", "cvw"]
    
    # loop through the dates
    for rad in rad_list:
        
        # store the multiprocess
        procs = []
        
        # loop through the radars
        for bm in range(24):

            if run_in_parallel:
                # cteate a process
                worker_kwargs = {"stm":stm, "etm":etm, "ftype":ftype,
				 "coords":coords,
                                 "config_filename":"../mysql_dbconfig_files/config.ini",
                                 "section":"midlat", "db_name":db_name}
                p = mp.Process(target=worker, args=(rad, bm),
                               kwargs=worker_kwargs)
                procs.append(p)
                
                # run the process
                p.start()
            else:
                worker(rad, bm, stm=stm, etm=etm, ftype=ftype,
		       coords=coords,
                       config_filename="../mysql_dbconfig_files/config.ini",
                       section="midlat", db_name=db_name)

        if run_in_parallel:
            # make sure the processes terminate
            for p in procs:
                p.join()

    return

if __name__ == "__main__":
    main(run_in_parallel=False)
