"""
written by Muhammad Rafiq, 2017-08-15
"""

import datetime as dt
from davitpy.pydarn.radar.radFov import slantRange, calcFieldPnt, calcAzOffBore
from davitpy.pydarn.radar.radStruct import site 
from davitpy.utils.coordUtils import coord_conv

import pdb


class latc_lonc_to_db(object):

    def __init__(self, rad, bmnum, stm, etm, coords="geo", ftype="fitacf"):

        """ calculates the center points of range-beam cells of a given radar beam 
        in geo coords and add them into the iscat db.

        Parameters
        ----------
        rad : str
            Three-letter radar code
        bmnum : int
            bmnum argument only works in search_allbeams is set to False
        stm : datetime.datetime
            The start time. 
        etm : datetime.datetime
            The end time. 
        coords : str
            Coordinate system, so far it should only be set to "geo"
        ftype : str
            SuperDARN file type 

        """ 

        rad_id_dict = {"bks":33, "wal":32, "cve":207, "cvw":206,
                       "fhe":205, "fhw":204, "ade":209, "adw":208,
                       "hok":40, "hkw":41, "tig":14, "unw":18, "bpk":24 }
        self.rad = rad
        self.rad_id = rad_id_dict[rad]
        self.bmnum = bmnum
        self.coords = coords
        self.stm = stm
        self.etm = etm
        self.table_name = self.rad + "_bm" + str(self.bmnum)
        self.ftype = ftype
        self.conn = self._create_dbconn()
        self.sites = self._create_site_list()

    def _create_dbconn(self,config_filename="../mysql_dbconfig_files/config.ini",
		       section="midlat", db_name=None):

	""" creates a db connection

        Parameters
        ----------
        config_filename: str
            name and path of the configuration file
        section: str, default to "midlat" 
            section of database configuration
        db_name : str, default to None
            Name of the MySQL db to which iscat data has been written

	"""

        from mysql.connector import MySQLConnection
        import sys
        sys.path.append("../")
        from mysql_dbutils.db_config import db_config
        import logging

        # read the db config info
        config =  db_config(config_filename=config_filename, section=section)
        config_info = config.read_db_config()

        # make db connection
        if db_name is None:
            db_name = self.rad + "_iscat_" + self.ftype
	try:
            conn = MySQLConnection(database=db_name, **config_info)
        except Exception, e:
            logging.error(e, exc_info=True)

        return conn

    def _create_site_list(self):

        """ creats a list of sites for a given self.rad for time the period between
        self.stm and self.etm """

        import sqlite3

        # create a sqlite3 db connection to the radar.sqlite3
        conn = sqlite3.connect(database="../data/sqlite3/radars.sqlite",
                               detect_types = sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()

        # select all the datetime values (tval) later than stm
        command = "SELECT tval FROM hdw WHERE id=? "
        command = '{:s}and tval>=? ORDER BY tval ASC'.format(command)
        cur.execute(command, (self.rad_id, self.stm))
        tvals_stm = cur.fetchall()
        tvals_stm = [x[0] for x in tvals_stm]

        # select all the datetime values (tval) later than etm
        command = "SELECT tval FROM hdw WHERE id=? "
        command = '{:s}and tval>=? ORDER BY tval ASC'.format(command)
        cur.execute(command, (self.rad_id, self.etm))
        tval_etm = cur.fetchone()[0]
        indx_etm = tvals_stm.index(tval_etm)

        # select the tvals of interest
        tvals = tvals_stm[:indx_etm+1]

        site_list = []
        for tval in tvals:
            site_list.append(site(code=self.rad, dt=tval))
        return site_list


    def add_latclonc_to_db(self):
        """ calculates latc and lonc of each range-beam cell in 'geo'
        coordinates and update them into the original table.
        If self.table_name does not exist in the db, it will not do anything"""

        import logging
        
        # check the db connection
        if not self.conn.is_connected():
            self.conn.reconnect()
        cur = self.conn.cursor(buffered=True)

        # check whether self.table_name exists. If not, do nothing
        command = "SHOW TABLES LIKE '{tb}'".format(tb=self.table_name)
        cur.execute(command)
        if not cur.fetchall():
            return

        # add new columns
        try:
            command ="ALTER TABLE {tb} ADD COLUMN geo_latc TEXT".format(tb=self.table_name) 
            cur.execute(command)
        except:
            # pass if the column geo_latc exists
            pass
        try:
            command ="ALTER TABLE {tb} ADD COLUMN geo_lonc TEXT".format(tb=self.table_name) 
            cur.execute(command)
        except:
            # pass if the column geo_lonc exists
            pass

        # iterate through tvals of the self.sites
        sdtm = self.stm
        for ii, st in enumerate(self.sites):
            if ii == len(self.sites)-1:
                edtm = self.etm
            else:
                edtm = st.tval

            # select data for the period between sdtm and edtm
            command = "SELECT slist, vel, frang, rsep, datetime " +\
                      "FROM {tb} WHERE datetime BETWEEN '{sdtm}' AND '{edtm}' "+\
                      "ORDER BY datetime"
            command = command.format(tb=self.table_name, sdtm=str(sdtm), edtm=str(edtm))
            try:
                cur.execute(command)
            except Exception, e:
                logging.error(e, exc_info=True)
            rows = cur.fetchall() 

            if rows != []:
                slist, vel, frang_old, rsep_old, date_time_old = rows[0]

                # calculate latc_all and lonc_all in 'geo' coords
                latc_all, lonc_all = calc_latc_lonc(self.sites[ii], self.bmnum, 
                                                    frang_old, rsep_old, 
                                                    altitude=300., elevation=None, coord_alt=0.,
                                                    coords="geo", date_time=None)

                # loop through rows 
                for row in rows:
                    slist, vel, frang, rsep, date_time = row
                    if (frang, rsep) != (frang_old, rsep_old):
                        # calculate latc_all and lonc_all in 'geo' coords if
                        # necessary (i.e., if frang or rsep changes)
                        latc_all, lonc_all = calc_latc_lonc(self.sites[ii],
                                                    self.bmnum, frang, rsep, 
                                                    altitude=300., elevation=None,
                                                    coord_alt=0.,
                                                    coords="geo", date_time=None)
                        
                        # update the _old values
                        frang_old, rsep_old = frang, rsep

                    # convert from string to float
                    slist = [int(float(x)) for x in slist.split(",")]
                    vel = [float(x) for x in vel.split(",")]

                    # exclude the slist values beyond maxgate and their correspinding velocities
                    vel = [vel[i] for i in range(len(vel)) if slist[i] < st.maxgate]
                    slist = [s for s in slist if s < st.maxgate]

                    # extract latc and lonc values
                    latc = [latc_all[s] for s in slist]
                    lonc = [lonc_all[s] for s in slist]

                    # convert to comma seperated text
                    slist = ",".join([str(x) for x in slist])
                    vel = ",".join([str(round(x,2)) for x in vel])
                    latc = ",".join([str(round(x,2)) for x in latc])
                    lonc = ",".join([str(round(x,2)) for x in lonc])

                    # update the table
                    command = "UPDATE {tb} SET slist='{slist}', vel='{vel}', " +\
                              "geo_latc='{latc}', geo_lonc='{lonc}' WHERE datetime = '{dtm}'"
                    command = command.format(tb=self.table_name, slist=slist, vel=vel,\
                                             latc=latc, lonc=lonc, dtm=date_time)
                    try:
                        cur.execute(command)
                    except Exception, e:
                        logging.error(e, exc_info=True)

            # update sdtm
            sdtm = edtm

        # commit the data into the db
        try:
            self.conn.commit()
        except Exception, e:
            logging.error(e, exc_info=True)

        # close db connection
        self.conn.close()
            
        return

def calc_latc_lonc(site, bmnum, frang, rsep, altitude=300.,
                   elevation=None, coord_alt=0., coords="geo",
                   date_time=None):

    """ calculates center lat and lon of all the range-gates of a given bmnum
    
    Parameters
    ----------
    site : davitpy.pydarn.radar.radStruct.site object
    bmnum : int
	bmnum argument only works in search_allbeams is set to False
    frang : int 
        Distance at which the zero range-gate starts [km]
    rsep : int
        Range seperation [km]
    altitude : float
        Default to 300. [km]
    elevation : float
        Defalut to None, in which case it will be estimated by the algorithm.
    coord_alt : float 
        like altitude, but only used for conversion from geographic to
        other coordinate systems.
        Default: 0, but set it to an appropriate float number for coord conversion
        at certain altitude.
    date_time : datetime.datetime
        the datetime for which the FOV is desired. Required for mag and mlt,
        and possibly others in the future. Default: None
    coords : str
        Coordinate system, should be set to "geo"

    Returns
    -------
    two lists
        Calculated center latitudes and longitudes of range gates of a given beam
    
    """
    import numpy as np

    # initialze papameters
    nbeams = site.maxbeam
    ngates = site.maxgate
    bmsep = site.bmsep
    recrise = site.recrise
    siteLat = site.geolat
    siteLon = site.geolon
    siteAlt = site.alt
    siteBore = site.boresite
    gates = np.arange(ngates)

    # Create output arrays
    lat_center = np.zeros(ngates, dtype='float')
    lon_center = np.zeros(ngates, dtype='float')

    # Calculate deviation from boresight for center of beam
    boff_center = bmsep * (bmnum - (nbeams - 1) / 2.0)

    # Calculate center slant range
    srang_center = slantRange(frang, rsep, recrise,
                              gates, center=True)

    # Calculate coordinates for Center of the current beam
    for ig in gates:
        talt = altitude
        telv = elevation
        t_c_alt = coord_alt

        # calculate projections
        latc, lonc = calcFieldPnt(siteLat, siteLon, siteAlt * 1e-3,
                                  siteBore, boff_center,
                                  srang_center[ig], elevation=telv,
                                  altitude=talt, model="IS",
                                  fov_dir="front")
        if(coords != 'geo'):
            lonc, latc = coord_conv(lonc, latc, "geo", coords,
                                    altitude=t_c_alt,
                                    date_time=date_time)

        # Save into output arrays
        lat_center[ig] = latc
        lon_center[ig] = lonc

    return lat_center, lon_center


def worker(rad, bmnum, stm, etm, ftype="fitacf"):

    import datetime as dt
    import sys

    # create a latc_lonc_to_db object
    t1 = dt.datetime.now()
    print("creating an latc_lonc_to_db object for beam " + str(bmnum) + " of " +\
          rad + " for period between " + str(stm) + " and " + str(etm))
    obj = latc_lonc_to_db(rad, bmnum, stm, etm, coords="geo", ftype=ftype)

    # calculate geolatc and geolonc and write them into a db where
    # iscat data is stored
    obj.add_latclonc_to_db()
    print("geolatc and geolonc have been written to db for beam " + str(bmnum) +\
           " of " + rad + " for period between " + str(stm) + " and " + str(etm))

    t2 = dt.datetime.now()
    print("Finishing an latc_lonc_to_db object for beam " + str(bmnum) +\
           " of " + rad + " for period between " + str(stm) + " and " +\
           str(etm) + " took " + str((t2-t1).total_seconds() / 60.) + " mins\n")
    
    return

def main(run_in_parallel=True):
    """ Call the functions above. Acts as an example code.
    Multiprocessing has been implemented to do parallel computing.
    A unit process is for a radar beam (i.e. a db table)"""

    import datetime as dt
    import multiprocessing as mp
    import sys
    sys.path.append("../")
    import logging

    # create a log file to which any error occured between client and
    # MySQL server communication will be written.
    logging.basicConfig(filename="./log_files/calc_geolatc_geolonc_hok.log",
                        level=logging.INFO)

    # input parameters
    stm = dt.datetime(2015, 1, 1)     # includes sdate
#    stm = dt.datetime(2011, 8, 1)     # includes sdate
    etm = dt.datetime(2017, 1, 1)     # does not include etm
    ftype = "fitacf"

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
                worker_kwargs = {"ftype":ftype}
		p = mp.Process(target=worker, args=(rad, bm, stm, etm),
                               kwargs=worker_kwargs)
		procs.append(p)

		# run the process
		p.start()

	    else:
                worker(rad, bm, stm, etm, ftype=ftype)

	if run_in_parallel:
	    # make sure the processes terminate
	    for p in procs:
		p.join()

    return

if __name__ == '__main__':
    main(run_in_parallel=True)
