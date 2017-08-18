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
            Coordinate system, should be set to "geo"
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
        section: str, default to mysql
            section of database configuration
        db_name : str, default to None
            Name of the MySQL db to which iscat data will be written

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
        coordinates and update them into the original table """
        
        # add new columns
        try:
            command ="ALTER TABLE {tb} ADD COLUMN latc TEXT".format(tb=self.table_name) 
            self.conn.cursor().execute(command)
        except:
            # pass if the column latc exists
            pass
        try:
            command ="ALTER TABLE {tb} ADD COLUMN lonc TEXT".format(tb=self.table_name) 
            self.conn.cursor().execute(command)
        except:
            # pass if the column lonc exists
            pass

        # iterate through tvals of the self.sites
        sdtm = self.stm
        for ii, st in enumerate(self.sites):
            if ii == len(self.sites)-1:
                edtm = self.etm
            else:
                edtm = st.tval
            command = "SELECT rowid, slist, vel, frang, rsep, datetime\
                       FROM {tb} WHERE (DATETIME(datetime)>'{sdtm}' and\
                       DATETIME(datetime)<='{edtm}') ORDER BY datetime".\
                       format(tb=self.table_name, sdtm=str(sdtm), edtm=str(edtm))
            self.conn.cursor().execute(command)
            rows = self.conn.cursor().fetchall() 
            if rows != []:
                rowid, slist, vel, frang_old, rsep_old, date_time_old = rows[0]

                # calculate latc_all and lonc_all in 'geo' coords
                latc_all, lonc_all = calc_latc_lonc(self.sites[ii], self.bmnum, frang_old, rsep_old, 
                                                    altitude=300., elevation=None, coord_alt=0.,
                                                    coords="geo", date_time=None)
                for row in rows:
                    rowid, slist, vel, frang, rsep, date_time = row
                    if (frang, rsep) != (frang_old, rsep_old):
                        latc_all, lonc_all = calc_latc_lonc(self.sites[ii], self.bmnum, frang, rsep, 
                                                    altitude=300., elevation=None, coord_alt=0.,
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
#                    command = "UPDATE {tb} SET slist='{slist}', vel='{vel}',\
#                               latc='{latc}', lonc='{lonc}' WHERE rowid=={rowid}".\
#                              format(tb=self.table_name, slist=slist, vel=vel,\
#                              latc=latc, lonc=lonc, rowid=rowid)
#                    self.conn.cursor.execute(command)

            # update sdtm
            sdtm = edtm

#        # commit the data into the db
#        self.conn.commit()

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
        other coordinate systems. Default: 0.
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

# test code
def main():

    # input parameters
    #rad_list = ["bks", "wal", "fhe", "fhw", "cve", "cvw"]
    rad_list = ["hok"]
    bmnum = 7
    ftype = "fitacf"

    stm = dt.datetime(2012,1,1)
    etm = dt.datetime(2012,2,29)

    objs = []
    for rad in rad_list:
        obj = latc_lonc_to_db(rad, bmnum, stm, etm, coords="geo", ftype=ftype)
        objs.append(obj)
    return objs
if __name__ == "__main__":
    objs = main()
    #objs.add_latclonc_to_db()
