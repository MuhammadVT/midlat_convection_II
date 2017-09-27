def cos_fit(input_table, output_table, db_name=None,
            config_filename="../mysql_dbconfig_files/config.ini",
            section="midlat", ftype="fitacf", coords="mlt",
            azbin_nvel_min=10, naz_min=3, az_span_min=30,
            sqrt_weighting=True):
    
    """ Does cosine fitting to the LOS data in each MLAT/MLT grid, 
    and stores the results in a different table named "master_cosfit_xxx". 
    This table has only the qualified latc-lonc grid points.

    Parameters
    ----------
    input_table : str
        A table name in db_name db
    output_table : str
        A table name in db_name db
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    db_name : str, default to None
        Name of the master db
    ftype : str
        SuperDARN file type
    coords : str
        Coordinates in which the binning process took place.
        Default to "mlt, can be "geo" as well. 
    azbin_nvel_min : int
        The minimum number of measurements an azimuthal bin should
        have to be qualified for cosfitting. 
    naz_min : int
        The minimum number of azimuthal bins within a grid cell.
        cosine fitting is done if a grid cell has at least
        naz_min number of qualified azimuthal bins
    az_span_min : int
        The minimum azimuhtal span a grid cell should have to
        be qualified for cosfitting.
    sqrt_weighting : bool
        if set to True, the fitting is weighted by the number of points within 
        each azimuthal bin. if set to False, all azimuthal bins are
        considered equal regardless of the nubmer of points
        each of them contains.

    Returns
    -------
    Nothing

    
    """

    import numpy as np
    import datetime as dt
    from mysql.connector import MySQLConnection
    import sys
    sys.path.append("../")
    from mysql_dbutils.db_config import db_config
    import logging

    # construct a db name
    if db_name is None:
        db_name = "master_" + coords + "_" +ftype

    # read db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make a connection to master db
    try:
        conn = MySQLConnection(database=db_name, **config_info)
        cur = conn.cursor(buffered=True)
    except Exception, e:
        logging.error(e, exc_info=True)

    # set output_table name
    if sqrt_weighting:
        output_table = output_table
    else:
        output_table = output_table + "_equal_weighting"

    # create output_table
    if coords == "mlt":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel_mag float(9,2)," +\
                  " vel_mag_err float(9,2)," +\
                  " vel_dir float(9,2)," +\
                  " vel_dir_err float(9,2)," +\
                  " vel_count INT," +\
                  " mag_gazmc_count SMALLINT," +\
                  " mag_glatc float(7,2)," +\
                  " mag_gltc float(8,2)," +\
                  " season VARCHAR(8), " +\
                  " CONSTRAINT cosfit_season PRIMARY KEY (" +\
                  "mag_glatc, mag_gltc, season))"
    elif coords == "geo":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel_mag float(9,2)," +\
                  " vel_mag_err float(9,2)," +\
                  " vel_dir float(9,2)," +\
                  " vel_dir_err float(9,2)," +\
                  " vel_count INT," +\
                  " geo_gazmc_count SMALLINT," +\
                  " geo_glatc float(7,2)," +\
                  " geo_gltc float(8,2)," +\
                  " season VARCHAR(8), " +\
                  " CONSTRAINT cosfit_season PRIMARY KEY (" +\
                  "geo_glatc, geo_gltc, season))"
    command = command.format(tb=output_table)
    try:
        cur.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)

    # select the velocity data grouping by latc-lonc bins.
    # Each latc-lonc cell should contain at lease naz_min gazmc bins 
    # with the azimuth span larger than az_span_min, and
    # with each gazmc bin having at least azbin_nvel_min number of 
    # measurements. 

    # add new columns
    if coords == "mlt":
        col_glatc = "mag_glatc"   # glatc -> gridded latitude center
        col_gltc = "mag_gltc"     # mlt hour in degrees
        col_gazmc = "mag_gazmc"   # gazmc -> gridded azimuthal center
        col_gazmc_count = "mag_gazmc_count"
    if coords == "geo":
        col_glatc = "geo_glatc"
        col_gltc = "geo_gltc"    # local time in degrees
        col_gazmc = "geo_gazmc"
        col_gazmc_count = "geo_gazmc_count"

    # set group_concat_max_len to a very large number so that large 
    # concatenated strings will not be truncated.
    command = "SET SESSION group_concat_max_len = 1000000"
    try:
        cur.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)

    # query  the data
    command = "SELECT count(*), {glatc}, {gltc}, group_concat({gazmc}), season FROM " +\
              "(SELECT * FROM {tb2} WHERE vel_count >= {azbin_nvel_min}) AS tbl " +\
              "GROUP BY {glatc}, {gltc}, season"
    command = command.format(tb2=input_table, glatc=col_glatc, gltc=col_gltc,
                             gazmc=col_gazmc, azbin_nvel_min=azbin_nvel_min)
    try:
        cur.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)
    rws = cur.fetchall()

    # filter out lat-lon grid points that have less than 3 qualified amimuthal bins 
    rws = [x for x in rws if x[0] >= naz_min]

    # filter out lat-lon grid points that have less than 30 degrees azimuthal span
    for rwi in rws:
        az_rwi = np.sort(np.array([int(x) for x in rwi[3].split(",")]))
        if len(az_rwi) == 3:
            if az_rwi.tolist()==[5, 345, 355] or az_rwi.tolist()==[5, 15, 355]:
                rws.remove(rwi)
            elif az_rwi.tolist()==[az_rwi[0], az_rwi[0]+10, az_rwi[0]+20]:
                rws.remove(rwi)
            else:
                continue
        else:
            continue

    azm_count = [x[0] for x in rws]
    lat = [x[1] for x in rws]
    lon = [x[2] for x in rws]
    seasons = [x[4] for x in rws]

    for ii in xrange(len(lat)):
        command = "SELECT vel_median, vel_count, {gazmc}, season FROM {tb2} " +\
                  "WHERE {glatc}={lat} "+ \
                  "AND {gltc}={lon} " +\
                  "AND season='{season}' " +\
                  "ORDER BY {gazmc}"
        command = command.format(tb2=input_table, glatc=col_glatc, gltc=col_gltc,
                                 season=seasons[ii], gazmc=col_gazmc,
                                 lat=lat[ii], lon=lon[ii])
        try:
            cur.execute(command)
        except Exception, e:
            logging.error(e, exc_info=True)
        rows = cur.fetchall()
        median_vel = np.array([x[0] for x in rows])
        vel_count = np.array([x[1] for x in rows])
        if sqrt_weighting:
            sigma =  1./np.sqrt(vel_count)
        else:
            sigma =  np.array([1.0 for x in rows])
        azm = np.array([x[2] for x in rows])

        # do cosine fitting with weight
        fitpars, perrs = cos_curve_fit(azm, median_vel, sigma)
        vel_mag = round(fitpars[0],2)
        vel_dir = round(np.rad2deg(fitpars[1]) % 360,1)
        vel_mag_err = round(perrs[0],2)
        vel_dir_err = round(np.rad2deg(perrs[1]) % 360, 1)

        # populate the out table 
        command = "INSERT IGNORE INTO {tb1} (vel_mag, "+\
                  "vel_mag_err, vel_dir, vel_dir_err, vel_count, "+\
                  "{gazmc_count_txt}, {glatc_txt}, {gltc_txt}, season) VALUES ({vel_mag}, "\
                  "{vel_mag_err}, {vel_dir}, {vel_dir_err}, {vel_count}, "+\
                  "{azmc_count}, {glatc}, {gltc}, '{season}')"
        command = command.format(tb1=output_table, gazmc_count_txt=col_gazmc_count,
                                 glatc_txt=col_glatc, gltc_txt=col_gltc, vel_mag=vel_mag,
                                 vel_mag_err=vel_mag_err, vel_dir=vel_dir,
                                 vel_dir_err=vel_dir_err, vel_count=np.sum(vel_count),
                                 azmc_count =azm_count[ii], glatc=lat[ii], gltc=lon[ii],
                                 season=seasons[ii])
        try:
            cur.execute(command)
        except Exception, e:
            logging.error(e, exc_info=True)
        print("finish inserting cosfit result at " +\
              str((lat[ii], lon[ii],seasons[ii])))

    # check whether database is still connected
    if not conn.is_connected():
        conn.reconnect()
    # commit the change
    try:
        conn.commit()
    except Exception, e:
        logging.error(e, exc_info=True)

    # close db connection
    conn.close()

    return

def cosfunc(x, Amp, phi):
    import numpy as np
    return Amp * np.cos(1 * x - phi)

def cos_curve_fit(azms, vels, sigma):
    import numpy as np
    from scipy.optimize import curve_fit
    fitpars, covmat = curve_fit(cosfunc, np.deg2rad(azms), vels, sigma=sigma)
    perrs = np.sqrt(np.diag(covmat)) 

    return fitpars, perrs

if __name__ == "__main__":
    
    import logging

    # create a log file to which any error occured between client and
    # MySQL server communication will be written.
    logging.basicConfig(filename="./log_files/master_cosfit_kp_00_to_23_hok_hkw.log",
                        level=logging.INFO)

    # initialize parameters
    input_table = "master_summary_hok_hkw_kp_00_to_23"
    output_table = "master_cosfit_hok_hkw_kp_00_to_23"
    cos_fit(input_table, output_table, db_name=None,
            config_filename="../mysql_dbconfig_files/config.ini",
            section="midlat", ftype="fitacf", coords="mlt",
            azbin_nvel_min=10, naz_min=3, az_span_min=30,
            sqrt_weighting=True)

