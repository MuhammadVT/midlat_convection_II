def imf_based_filter(imf_table, output_table, stm, etm,
		     ftype="fitacf", coords="mlt",
                     kp_lim=[0.0, 9.0],
		     config_filename="../mysql_dbconfig_files/config.ini",
		     section="midlat",
                     input_dbname=None, input_table=None,
		     output_dbname=None,
                     imf_dbname=None, imf_db_location="../../data/sqlite3/"):

    """ Selects the data based on IMF.

    Parameters
    ----------
    stm : datetime.datetime
	The start time
    etm : datetime.datetime
	The end time
    imf_table : list
        The binned imf table from which binned imf data will be taken.
    output_table : str
        Name of the table where filtered iscat data will be stored.
    ftype : str
        SuperDARN LOS data file type
    coords : str
        Coordinate systems. valid inputs are "geo" or "mlt".
    kp_lim : list
        The range of Kp. The range bondaries are inclusive.
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    input_dbname : str
        Name of the input db.
        Default to None.
    input_table : str
        Name of the table where the iscat data will be extracted.
    output_dbname : str
        Name of the output db.
        Default to None.
    imf_dbname : str
        Name of the sqlite3 db where imf data are stored.
        Default to None.
    imf_db_location : str
        The path to imf_dbname sqlite db.
        Default to None.

    Returns
    -------
    Nothing
    
    """

    import datetime as dt
    import sqlite3
    import os
    import sys
    sys.path.append("../../")
    from mysql.connector import MySQLConnection
    from mysql_dbutils.db_config import db_config
    from mysql_dbutils import db_tools
    import logging

    # construct db names and table names
    if input_dbname is None:
        input_dbname = "master_" + coords + "_" + ftype
    if output_dbname is None:
        output_dbname = "master_" + coords + "_" + ftype + "_binned_by_imf_clock_angle"
    if imf_dbname is None:
        imf_dbname = "binned_imf.sqlite"

    # create the output db (if not exist) 
    try:
        # create a db
        db_tools.create_db(output_dbname, config_filename=config_filename)
    except Exception, e:
        logging.error(e, exc_info=True)

    # make db connection to binned imf data
    conn_imf = sqlite3.connect(database =imf_db_location + imf_dbname,
                               detect_types=sqlite3.PARSE_DECLTYPES)
    cur_imf = conn_imf.cursor()

    # make db connection for master data
    # read the db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make a connection
    try:
        conn_input = MySQLConnection(database=input_dbname, **config_info)
        cur_input = conn_input.cursor(buffered=True)
    except Exception, e:
        logging.error(e, exc_info=True)

    # make a connection
    try:
        conn_output = MySQLConnection(database=output_dbname, **config_info)
        cur_output = conn_output.cursor(buffered=True)
    except Exception, e:
        logging.error(e, exc_info=True)

    # select the time intervals based on Kp 
    command = "SELECT datetime FROM {tb} " +\
	      "WHERE (datetime BETWEEN '{stm}' AND '{etm}') AND " +\
	      "(kp BETWEEN {kp_min} AND {kp_max}) ORDER BY datetime"
    command = command.format(tb=imf_table, stm=stm, etm=etm,
			     kp_min=kp_lim[0],kp_max=kp_lim[1])
    cur_imf.execute(command)
    dtms = cur_imf.fetchall()
    dtms = [x[0] for x in dtms]

    # close conn_imf opened for imf db
    conn_imf.close()

    # create the output table where the imf filtered master iscat data will be stored
    if coords == "mlt":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel float(9,2)," +\
                  " mag_glatc float(7,2)," +\
                  " mag_gltc float(8,2)," +\
                  " mag_gazmc SMALLINT," +\
                  " datetime DATETIME, " +\
                  " season VARCHAR(8), " +\
                  " rad VARCHAR(3), " +\
                  " CONSTRAINT imf_filtered PRIMARY KEY (" +\
                  "mag_glatc, mag_gltc, mag_gazmc, datetime, rad))"
        command_insert = "INSERT IGNORE INTO {tb2} (vel, mag_glatc, mag_gltc, " +\
                  "mag_gazmc, datetime, season, rad) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    elif coords == "geo":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel float(9,2)," +\
                  " geo_glatc float(7,2)," +\
                  " geo_gltc float(8,2)," +\
                  " geo_gazmc SMALLINT," +\
                  " datetime DATETIME, " +\
                  " season VARCHAR(8), " +\
                  " rad VARCHAR(3), " +\
                  " CONSTRAINT imf_filtered PRIMARY KEY (" +\
                  "geo_glatc, geo_gltc, geo_gazmc, datetime, rad))"
        command_insert = "INSERT IGNORE INTO {tb2} (vel, geo_glatc, geo_gltc, " +\
                  "geo_gazmc, datetime, season, rad) VALUES (%s, %s, %s, %s, %s, %s, %s)"

    command = command.format(tb=output_table)
    command_insert = command_insert.format(tb2=output_table)
    try:
        cur_output.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)

    # filter iscat db based on imf 
    # loop through every xx-minute of imf table interval
#    for dtm in dtms:
#        print("Filtering data for {dtm}".format(dtm=str(dtm)))
#        command = "SELECT * FROM {tb} WHERE datetime = '{dtm}'"
#        command = command.format(tb=input_table, dtm=dtm)
#
#        # check the db connection before fetching 
#        if not conn_input.is_connected():
#            conn_input.reconnect()
#
#        # fetch the data
#        try:
#	    cur_input.execute(command)
#	    rows = cur_input.fetchall()
#        except Exception, e:
#            logging.error(e, exc_info=True)
#        
#        # insert the data into the output table
#        if rows:
#            for rw in rows:
#                vel, lat, lt, azm, dtm, season, rad = rw
#
#                # check the db connection before inserting
#                if not conn_output.is_connected():
#                    conn_output.reconnect()
#                # insert the data
#                try:
#                    cur_output.execute(command_insert, (vel, lat, lt, azm, dtm, season, rad))
#                except Exception, e:
#                    logging.error(e, exc_info=True)
    batch_size = 1000
    niter = len(dtms) / batch_size + 1 
    for n in range(niter):
        dtms_n = dtms[n * batch_size : batch_size * (n+1)]
	command = "SELECT * FROM {tb} " +\
		  "WHERE (datetime BETWEEN '{stm}' AND '{etm}') AND " +\
		  "(datetime IN {dtms_n})"
	command = command.format(tb=input_table, stm=stm, etm=etm,
				 dtms_n=tuple([str(x) for x in dtms_n]))

	# check the db connection before fetching 
	if not conn_input.is_connected():
	    conn_input.reconnect()

	# fetch the data
	try:
	    print("Fetching data from " + input_table + " for interval between " +\
                  str(dtms_n[0]) + " and " + str(dtms_n[-1]))
	    cur_input.execute(command)
	    rows = cur_input.fetchall()
	except Exception, e:
	    logging.error(e, exc_info=True)
	
	# insert the data into the output table
	if rows:
            print("Inserting data to " + output_table + " for interval between " +\
                  str(dtms_n[0]) + " and " + str(dtms_n[-1]))
	    for rw in rows:
		vel, lat, lt, azm, dtm, season, rad = rw

		# check the db connection before inserting
		if not conn_output.is_connected():
		    conn_output.reconnect()
		# insert the data
		try:
		    cur_output.execute(command_insert, (vel, lat, lt, azm, dtm, season, rad))
		except Exception, e:
		    logging.error(e, exc_info=True)


    # check the db connection before committing
    if not conn_output.is_connected():
        conn_output.reconnect()
    # commit the change
    try:
        conn_output.commit()
    except Exception, e:
        logging.error(e, exc_info=True)

    # close the connections
    conn_input.close()
    conn_output.close()

    return

def worker(imf_table, output_table, stm, etm,
	   ftype="fitacf", coords="mlt",
           kp_lim=[0.0, 9.0],
	   config_filename="../../mysql_dbconfig_files/config.ini",
	   section="midlat",
           input_dbname=None, input_table=None,
	   output_dbname=None,
           imf_dbname=None, imf_db_location="../../data/sqlite3/"):
    """ A worker function for parallel computing"""

    # filter the data
    print("Start filtering data from " + imf_table)
    t1 = dt.datetime.now()
    imf_based_filter(imf_table, output_table, stm, etm,
		     ftype=ftype, coords=coords,
                     kp_lim=kp_lim,
		     config_filename="../../mysql_dbconfig_files/config.ini",
		     section="midlat",
                     input_dbname=input_dbname,
                     input_table=input_table,
                     output_dbname=output_dbname,
                     imf_dbname=imf_dbname,
                     imf_db_location=imf_db_location)
    t2 = dt.datetime.now()
    print("finish filtering data based on " + imf_table +\
          ", which took " + str((t2-t1).total_seconds() / 60.) + " mins\n")

    return

def main(run_in_parallel=False):
    """ Executes the above functions in parallel.
    Unit process runs for an imf_bin"""

    import multiprocessing as mp
    import logging
    import datetime as dt
    import numpy as np

    stm = dt.datetime(2011, 1, 1)
    etm = dt.datetime(2017, 1, 1)

    # initialize parameters
    coords = "mlt"
    ftype="fitacf"

    kp_lim = [0.0, 9.0]    # Used for filtering imf data in imf_table based on kp,
			   # which is different from the kp values in kp_text below.
			   # The range boundaries are inclusive
    rads_txt = "six_rads" 
    kp_text = "_kp_00_to_23"

    input_dbname = "master_" + coords + "_" + ftype
    input_table = "master_" + rads_txt + kp_text
    output_dbname = "master_" + coords + "_" + ftype + "_binned_by_imf_clock_angle"
    imf_dbname="binned_imf.sqlite"
    imf_db_location="../../data/sqlite3/"

    # set the imf bins
    sector_width = 60
    sector_center_dist = 90
    imf_bins = [[x-sector_width/2, x+sector_width/2] for x in np.arange(0, 360, sector_center_dist)]

    bvec_max = 0.85
    before_mins=80
    after_mins=0
    del_tm=10

    # create a log file to which any error occured between client and
    # MySQL server communication will be written.
    logging.basicConfig(filename="../log_files/master_" + rads_txt +\
                                 kp_text + "_binned_by_imf_clock_angle.log",
                        level=logging.INFO)
    
    # run in parallel
    # store the multiprocess
    procs = []
    # loop through radars
    for imf_bin in imf_bins: 
        imf_table = "b" + str((imf_bin[0]%360)) + "_b" + str(imf_bin[1]%360) + \
                     "_before" + str(before_mins) + "_after" +  str(after_mins) + \
                     "_bvec" + str(bvec_max).split('.')[-1]

	output_table = "master_" + rads_txt + kp_text +\
		       "_b" + str((imf_bin[0]%360)) + "_b" + str(imf_bin[1]%360) +\
		       "_bfr" + str(before_mins) +\
		       "_aftr" +  str(after_mins) +\
		       "_bvec" + str(bvec_max).split('.')[-1]

        if run_in_parallel:
            # cteate a process
            worker_kwargs = {"ftype":ftype, "coords":coords,
			     "kp_lim": kp_lim,
                             "config_filename":"../../mysql_dbconfig_files/config.ini",
                             "section":"midlat",
			     "input_dbname": input_dbname,
			     "input_table": input_table,
			     "output_dbname": output_dbname,
			     "imf_dbname": imf_dbname,
			     "imf_db_location": imf_db_location}
            p = mp.Process(target=worker, args=(imf_table, output_table, stm, etm),
                           kwargs=worker_kwargs)
            procs.append(p)
            
            # run the process
            p.start()
            
        else:
            # run in serial
	    worker(imf_table, output_table, stm, etm,
		   ftype=ftype, coords=coords,
		   kp_lim=kp_lim,
		   config_filename="../../mysql_dbconfig_files/config.ini",
		   section="midlat", 
		   input_dbname=input_dbname,
		   input_table=input_table,
		   output_dbname=output_dbname,
		   imf_dbname=imf_dbname,
		   imf_db_location=imf_db_location)
            
    if run_in_parallel:
        # make sure the processes terminate
        for p in procs:
            p.join()

if __name__ == "__main__":
    import datetime as dt
    t1 = dt.datetime.now()
    main(run_in_parallel=False)
    t2 = dt.datetime.now()
    print("Finishing imf_based filtering took " +\
    str((t2-t1).total_seconds() / 60.) + " mins\n")


