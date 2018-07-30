def gmi_based_filter(rad, output_table, stm=None, etm=None, ftype="fitacf", coords="mlt",
                     isKp_based=True, kp_lim=[0.0, 2.3],
                     isSymH_based=False, symh_min=-50,
		     config_filename="../mysql_dbconfig_files/config.ini",
		     section="midlat",
                     ten_min_median_dbname=None,
                     gmi_dbname=None, gmi_db_location="../../data/sqlite3/"):

    """ Selects the data based on geomagnetic indicies.

    Parameters
    ----------
    rads : str
        Three-letter name of a radar whose data will be filtered.
    output_table : str
        Name of the table where filtered iscat data will be stored.
    stm : datetime.datetime
        The start time.
        Default to None, in which case takes the earliest in db.
    etm : datetime.datetime
        The end time.
        Default to None, in which case takes the latest time in db.
        NOTE: if stm is None then etm should also be None, and vice versa.
    ftype : str
        SuperDARN LOS data file type
    coords : str
        Coordinate systems. valid inputs are "geo" or "mlt".
    isKp_based : bool
        If set to True, data will be filtered based on Kp.
    kp_lim : list
        The range of Kp. The range bondaries are inclusive.
    isSymH_based : bool
        If set to True, data will be filtered based on SymH.
    symh_min : float
        The lower limit for SYMH.
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    ten_min_median_dbname : str
        Name of the db where ten min median iscat data is stored.
        Default to None.
    gmi_dbname : str
        Name of the sqlite3 db where geomagnetic indices are stored.
        Default to None.
    gmi_db_location : str
        The path to gmi_dbname sqlite db.
        Default to None.

    Returns
    -------
    Nothing

    Note
    ----
    SymH base filter has not been implemented yet
    
    """

    import datetime as dt
    import sqlite3
    import os
    import sys
    sys.path.append("../../")
    from mysql.connector import MySQLConnection
    from mysql_dbutils.db_config import db_config
    import logging

    # construct db names and table names
    if ten_min_median_dbname is None:
        ten_min_median_dbname = "ten_min_median_" + coords + "_" + ftype
    if gmi_dbname is None:
        gmi_dbname = "gmi_imf.sqlite"
    input_table = rad + "_" + ftype

    # make db connection to geomag indicies data
    conn_gmi = sqlite3.connect(database =gmi_db_location + gmi_dbname,
                               detect_types=sqlite3.PARSE_DECLTYPES)
    cur_gmi = conn_gmi.cursor()

    # make db connection for ten-min median filtered iscat data
    # read the db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make a connection
    try:
        conn_iscat = MySQLConnection(database=ten_min_median_dbname, **config_info)
        cur_iscat = conn_iscat.cursor(buffered=True)
    except Exception, e:
        logging.error(e, exc_info=True)

    # select the time intervals base on SymH
    if isSymH_based:
        pass

    # select the time intervals based on Kp 
    if isKp_based:
        tbl_nm = "kp"
        # do the convertion to the data between stm and etm
        if (stm is not None) and (etm is not None):
	    command = "SELECT datetime FROM {tb} WHERE (datetime BETWEEN '{sdtm}' AND '{edtm}') " +\
		      "AND (kp BETWEEN {kp_min} AND {kp_max})"
	    command = command.format(tb=tbl_nm, sdtm=stm, edtm=etm,
				     kp_min=kp_lim[0],kp_max=kp_lim[1])

        # do the convertion to the data between stm and etm if any of them is None
	else:
	    command = "SELECT datetime FROM {tb} WHERE kp BETWEEN {kp_min} AND {kp_max}"
	    command = command.format(tb=tbl_nm, kp_min=kp_lim[0],kp_max=kp_lim[1])
        cur_gmi.execute(command)
        dtms = cur_gmi.fetchall()
        dtms = [x[0] for x in dtms]

    # close conn_gmi opened for gmi db
    conn_gmi.close()

    # create the output table where the filtered iscat data will be stored
    if coords == "mlt":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel float(9,2)," +\
                  " mag_glatc float(7,2)," +\
                  " mag_gltc float(8,2)," +\
                  " mag_gazmc SMALLINT," +\
                  " datetime DATETIME, " +\
                  " rad VARCHAR(3), " +\
                  " CONSTRAINT filtered PRIMARY KEY (" +\
                  "mag_glatc, mag_gltc, mag_gazmc, datetime, rad))"
        command_insert = "INSERT IGNORE INTO {tb2} (vel, mag_glatc, mag_gltc, " +\
                  "mag_gazmc, datetime, rad) VALUES (%s, %s, %s, %s, %s, %s)"
    elif coords == "geo":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel float(9,2)," +\
                  " geo_glatc float(7,2)," +\
                  " geo_gltc float(8,2)," +\
                  " geo_gazmc SMALLINT," +\
                  " datetime DATETIME, " +\
                  " rad VARCHAR(3), " +\
                  " CONSTRAINT filtered PRIMARY KEY (" +\
                  "geo_glatc, geo_gltc, geo_gazmc, datetime, rad))"
        command_insert = "INSERT IGNORE INTO {tb2} (vel, geo_glatc, geo_gltc, " +\
                  "geo_gazmc, datetime, rad) VALUES (%s, %s, %s, %s, %s, %s)"

    command = command.format(tb=output_table)
    command_insert = command_insert.format(tb2=output_table)
    try:
        cur_iscat.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)

    # filter iscat db based on kp
    # loop through every 3 hour interval of kp
    for dtm in dtms:
        sdtm = dtm
        edtm = sdtm + dt.timedelta(hours=3)
        print("Filtering data for time interval between {sdtm} and {edtm}".\
                format(sdtm=str(sdtm), edtm=str(edtm)))
        command = "SELECT * FROM {tb} WHERE datetime BETWEEN '{sdtm}' " +\
                  "AND '{edtm}'"
        command = command.format(tb=input_table, sdtm=sdtm, edtm=edtm)

        # check the db connection before fetching 
        if not conn_iscat.is_connected():
            conn_iscat.reconnect()

        # fetch the data
        try:
	    cur_iscat.execute(command)
	    rows = cur_iscat.fetchall()
        except Exception, e:
            logging.error(e, exc_info=True)
        
        # insert the data into a table
        if rows:
            for rw in rows:
                vel, lat, lt, azm, dtm = rw

                # check the db connection before inserting
                if not conn_iscat.is_connected():
                    conn_iscat.reconnect()
                # insert the data
                try:
                    cur_iscat.execute(command_insert, (vel, lat, lt, azm, dtm, rad))
                except Exception, e:
                    logging.error(e, exc_info=True)

    # check the db connection before committing
    if not conn_iscat.is_connected():
        conn_iscat.reconnect()
    # commit the change
    try:
        conn_iscat.commit()
    except Exception, e:
        logging.error(e, exc_info=True)

    # close the connections
    conn_iscat.close()

    return

def worker(rad, output_table, stm=None, etm=None, ftype="fitacf", coords="mlt",
           isKp_based=True, kp_lim=[0.0, 2.3],
           isSymH_based=False, symh_min=-50,
	   config_filename="../../mysql_dbconfig_files/config.ini",
	   section="midlat",
           ten_min_median_dbname=None,
           gmi_dbname=None, gmi_db_location="../../data/sqlite3/"):
    """ A worker function for parallel computing"""

    # filter the data
    print("Start filtering data from " + rad)
    t1 = dt.datetime.now()
    gmi_based_filter(rad, output_table, stm=stm, etm=etm, ftype=ftype, coords=coords,
                     isKp_based=isKp_based, kp_lim=kp_lim,
                     isSymH_based=isSymH_based, symh_min=symh_min,
		     config_filename="../../mysql_dbconfig_files/config.ini",
		     section="midlat",
                     ten_min_median_dbname=ten_min_median_dbname,
                     gmi_dbname=gmi_dbname,
                     gmi_db_location=gmi_db_location)
    t2 = dt.datetime.now()
    print("finish filtering data from " + rad +\
          ", which took " + str((t2-t1).total_seconds() / 60.) + " mins\n")

    return

def main(run_in_parallel=False):
    """ Executes the above functions in parallel.
    Unit process runs for a single radar"""

    import multiprocessing as mp
    import logging

    # initialize parameters
    #stm = None
    #etm = None
    stm = dt.datetime(2017, 1, 1)
    etm = dt.datetime(2018, 7, 1)

    #rad_list = ["hok", "hkw"]
    #rad_list = ["ade", "adw"]
    rad_list = ['bks', 'wal', 'fhe', 'fhw', 'cve', 'cvw'] 

    kp_lim = [0.0, 2.3]    # the range boundaries are inclusive
    #kp_lim = [0.0, 1.3]    # the range boundaries are inclusive
    #kp_lim = [0.0, 0.3]    # the range boundaries are inclusive
    #kp_lim = [0.7, 1.3]    # the range boundaries are inclusive
    #kp_lim = [1.7, 2.3]    # the range boundaries are inclusive
    #kp_lim = [2.7, 3.3]    # the range boundaries are inclusive
    #kp_lim = [2.7, 4.3]    # the range boundaries are inclusive
    #kp_lim = [3.7, 9.0]    # the range boundaries are inclusive
    kp_text = "_to_".join(["".join(str(x).split(".")) for x in kp_lim])
	
    coords = "mlt"
    ftype="fitacf"
    #output_table = "_".join(rad_list) + "_kp_" + kp_text + "_" + ftype
    output_table = "six_rads_kp_" + kp_text + "_" + ftype
    isKp_based=True
    isSymH_based=False
    symh_min=-50
    ten_min_median_dbname=None
    gmi_dbname=None
    gmi_db_location="../../data/sqlite3/"

    # create a log file to which any error occured between client and
    # MySQL server communication will be written.
    logging.basicConfig(filename="../log_files/six_rads_kp_" + kp_text + ".log",
                        level=logging.INFO)
    
    # run in parallel
    # store the multiprocess
    procs = []
    # loop through radars
    for rad in rad_list: 
        if run_in_parallel:
            # cteate a process
            worker_kwargs = {"stm":stm, "etm":etm, "ftype":ftype, "coords":coords,
			     "isKp_based": isKp_based, "kp_lim": kp_lim,
			     "isSymH_based": isSymH_based, "symh_min": symh_min,
                             "config_filename":"../../mysql_dbconfig_files/config.ini",
                             "section":"midlat",
			     "ten_min_median_dbname": ten_min_median_dbname,
			     "gmi_dbname": gmi_dbname,
			     "gmi_db_location": gmi_db_location}
            p = mp.Process(target=worker, args=(rad, output_table),
                           kwargs=worker_kwargs)
            procs.append(p)
            
            # run the process
            p.start()
            
        else:
            # run in serial
	    worker(rad, output_table, stm=stm, etm=etm, ftype=ftype, coords=coords,
			 isKp_based=isKp_based, kp_lim=kp_lim,
			 isSymH_based=isSymH_based, symh_min=symh_min,
			 config_filename="../../mysql_dbconfig_files/config.ini",
			 section="midlat", 
			 ten_min_median_dbname=ten_min_median_dbname,
			 gmi_dbname=gmi_dbname,
			 gmi_db_location=gmi_db_location)
            
    if run_in_parallel:
        # make sure the processes terminate
        for p in procs:
            p.join()

if __name__ == "__main__":
    import datetime as dt
    t1 = dt.datetime.now()
    main(run_in_parallel=True)
    t2 = dt.datetime.now()
    print("Finishing gmi_based filtering took " +\
    str((t2-t1).total_seconds() / 60.) + " mins\n")


