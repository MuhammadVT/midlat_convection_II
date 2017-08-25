def ten_min_median(rad, stm, etm, ftype="fitacf", 
		   coords="mlt",
		   config_filename="../mysql_dbconfig_files/config.ini",
		   section="midlat", iscat_dbname=None, output_dbname=None)
    
    """ bins ten minutes of data from a single radar by choosing 
    the median vector in each azimuth bin within each grid cell. 
    The results are stored in a different db file that combines all the beams's 
    data into a single table named by the radar.

    Parameters
    ----------
    rad : str
        Three-letter radar code
    stm : datetime.datetime
        The start time.
    etm : datetime.datetime
        The end time.
    ftype : str
        SuperDARN file type
    coords : str
        Coordinates in which the binning process takes place.
        Default to "mlt. Can be "geo" as well. 
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    iscat_dbname : str, default to None
        Name of the MySQL db to which iscat data has been written
    output_dbname : str, default to None
        Name of the MySQL db to which ten-min median will be written

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

    # read the db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make a connection to iscat db
    if iscat_dbname is None:
        iscat_dbname = rad + "_iscat_" + ftype
    try:
        conn_iscat = MySQLConnection(database=iscat_dbname, **config_info)
    except Exception, e:
        logging.error(e, exc_info=True)
    cur_iscat = conn_iscat.cursor(buffered=True)

    # make a connection to ten-min-median db
    if output_dbname is None:
        output_dbname = "ten_min_median_" + coords + "_" + ftype
    try:
        conn_output = MySQLConnection(database=output_dbname, **config_info)
    except Exception, e:
        logging.error(e, exc_info=True)
    cur_output = conn_output.cursor(buffered=True)

    # create a table that combines all the beams of a radar
    if coords == "mlt":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel float(9,2) DEFAULT NULL," +\
                  " mag_glatc float(7,2) DEFAULT NULL," +\
                  " mag_gltc float(8,2) DEFAULT NULL," +\
                  " mag_gazmc TINYINT(5) DEFAULT NULL," +\
                  " datetime DATETIME DEFAULT NULL)"
    elif coords == "geo":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel float(9,2) DEFAULT NULL," +\
                  " geo_glatc float(7,2) DEFAULT NULL," +\
                  " geo_gltc float(8,2) DEFAULT NULL," +\
                  " geo_gazmc TINYINT(5) DEFAULT NULL," +\
                  " datetime DATETIME DEFAULT NULL)"
    command = command.format(tb=rad)
    cur_output.execute(command)

    # get all the table names in iscat db
    cur_iscat.execute("SHOW TABLES")
    tbl_names = cur_iscat.fetchall()
    tbl_names = [x[0] for x in tbl_names]

    # lengh of the time interval during which the data is median filtered 
    len_tm = 10    # minutes

    # initial starting and ending time of the time interval given by len_tm
    sdtm = stm
    edtm = sdtm + dt.timedelta(minutes=len_tm)

    # slide the ten-minute window from stm to etm
    while edtm <= etm:
        # bin_vel stores the velocity data as {glatc-glonc-gazmc: [velocites]}  
        bin_vel = {}
        for tbl in tbl_names:
            # select column variables from iscat for a ten-minute interval
            if coords == "mlt":
                command = "SELECT vel, mag_glatc, mag_gltc, mag_gazmc " +\
                          "FROM {tb} WHERE datetime BETWEEN '{sdtm}' AND '{edtm}'"
            elif coords == "geo":
                command = "SELECT vel, geo_glatc, geo_gltc, geo_gazmc " +\
                          "FROM {tb} WHERE datetime BETWEEN '{sdtm}' AND '{edtm}'"
            command = command.format(tb=tbl, sdtm=sdtm, edtm=edtm)
            try:
                cur_iscat.execute(command)
            except Exception, e:
                logging.error(e, exc_info=True)
            rows_tmp = cur_iscat.fetchall()

            if rows_tmp:
                # loop through each row
                for row in rows_tmp:
                    vel, lat, lon, az = row
                    
                    if None not in row:
                        # convert from string to float
                        vel = [float(x) for x in vel.split(",")]
                        lat = [float(x) for x in lat.split(",")]
                        lon = [float(x) for x in lon.split(",")]
                        az = [float(x) for x in az.split(",")]

                        for i in range(len(vel)):
                            # exclude NaN elements
                            if np.isnan(lat[i]):
                                continue

                            # build teh bin_vel dict ({glatc-glonc-gazmc: [velocites]})
                            try:
                                bin_vel[(lat[i],lon[i],az[i])].append(vel[i])
                            except KeyError:
                                bin_vel[(lat[i],lon[i],az[i])] = [vel[i]]
                    else:
                        continue
            
            else:
                continue

        if bin_vel:
            # take the mid-point of sdtm and edtm
            mid_tm = sdtm + dt.timedelta(minutes=len_tm/2.)
                
            # populate the rad table 
            for ky in bin_vel.keys(): 
                # take the median value
                bin_vel[ky] = round(np.median(bin_vel[ky]),2)
                command = "INSERT INTO {tb} (vel, glatc, glonc, gazmc, datetime) " +\
                          "VALUES (?, ?, ?, ?, ?)"
                command = command.format(tb=rad+ "_" + ftype)
                try:
                    cur_output.execute(command, (bin_vel[ky], ky[0], ky[1], ky[2], mid_tm))
                except Exception, e:
                    logging.error(e, exc_info=True)

        print "finished taking the median velocity between " + str(sdtm)\
                + " and " + str(edtm) + " of " + rad

        # update starting and ending time of the time interval given by len_tm
        sdtm = edtm
        edtm = sdtm + dt.timedelta(minutes=len_tm)

    # commit the change
    try:
        conn_output.commit()
    except Exception, e:
        logging.error(e, exc_info=True)

    # close db connections
    conn_iscat.close()
    conn_output.close()

    return

def worker(rad, stm, etm, ftype="fitacf", 
           coords="mlt", config_filename="../mysql_dbconfig_files/config.ini",
           section="midlat", iscat_dbname=None, output_dbname=None)

    import datetime as dt

    # input parameters
    ftype = "fitacf"

    # loop through time interval
    for dd in range(len(stms)):
        stm = stms[dd]
        etm = etms[dd]
                
        # loop through radars
        for rad in rads:

            # take ten minutes median values
            print("start working on table " + rad + " for interval between " +\
                  str(stm) + " and " + str(etm))
            ten_min_median(rad, stm, etm, ftype=ftype, 
                           coords=coords,
                           config_filename=config_filename,
                           section=section, iscat_dbname=iscat_dbname,
                           output_dbname=output_dbname)
            print("finish taking ten mimute median filtering on " + rad +\
                  " for interval between " + str(stm) + " and " + str(etm))

    return

def main(run_in_parallel=True):
    """ Call the functions above. Acts as an example code.
    Multiprocessing has been implemented to do parallel computing.
    A unit process runs for a radar (i.e. a radar data is written to
    a db table named with rad + ftype)"""

    import datetime as dt
    import multiprocessing as mp
    import logging
    
    # create a log file to which any error occured between client and
    # MySQL server communication will be written.
    logging.basicConfig(filename="./log_files/ten_min_median_hok.log",
                        level=logging.INFO)
    # input parameters
    stm = dt.datetime(2011, 1,1) 
    etm = dt.datetime(2017, 1,1) 
    ftype = "fitacf"
    iscat_dbname = None       # if set to None, default iscat db would be used. 
    output_dbname = None       # if set to None, default ten_min_median db would be used. 
    
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
	if run_in_parallel:
	    # run in parellel
	    # cteate a process
	    worker_kwargs = {"ftype":ftype, "coords":coords,
			     "config_filename":"../mysql_dbconfig_files/config.ini",
			     "section":"midlat", "iscat_dbname":iscat_dbname,
			     "output_dbname":output_dbname}
	    p = mp.Process(target=worker, args=(rad, stm, etm),
			   kwargs=worker_kwargs)
	    procs.append(p)
	    
	    # run the process
	    p.start()
	    
	else:
	    # run in serial
            worker(rad, stm, etm, ftype=ftype, 
                           coords=coords,
                           config_filename=config_filename,
                           section=section, iscat_dbname=iscat_dbname,
                           output_dbname=output_dbname)
            
        if run_in_parallel:
            # make sure the processes terminate
            for p in procs:
                p.join()

    return

if __name__ == "__main__":
    main(run_in_parallel=False)
