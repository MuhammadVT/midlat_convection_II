def build_master_table(input_table, output_table, ftype="fitacf", coords="mlt",
                       config_filename="../mysql_dbconfig_files/config.ini",
                       section="midlat", input_dbname=None, output_dbname=None):
   
    """ combines all the ten-min median filtered gridded iscat data into
    one master table. 
    The results are stored in a different db file.

    Parameters
    ----------
    input_table : str
        A table name from input_dbname
    output_table : str
        A table name from output_dbname
    ftype : str
        SuperDARN file type
    coords : str
        Coordinates in which the binning process took place.
        Default to "mlt, can be "geo" as well. 
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    input_dbname : str, default to None
        Name of the MySQL db where ten-min median data are stored.
    output_dbname : str, default to None
        Name of the master db

    Returns
    -------
    Nothing

    """

    import numpy as np
    import datetime as dt
    from mysql.connector import MySQLConnection
    from month_to_season import get_season_by_month
    import sys
    sys.path.append("../")
    from mysql_dbutils.db_config import db_config
    from mysql_dbutils import db_tools
    import logging

    # create db name
    if input_dbname is None:
        input_dbname = "ten_min_median_" + coords + "_" + ftype
    if output_dbname is None:
        output_dbname = "master_" + coords + "_" +ftype

    # create a db (if not exist) that combines all the data
    try:
        # create a db
        db_tools.create_db(output_dbname)
    except Exception, e:
        logging.error(e, exc_info=True)

    # read db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make a connection to ten-min median iscat db
    try:
        conn_ten = MySQLConnection(database=input_dbname, **config_info)
        cur_ten = conn_ten.cursor(buffered=True)
    except Exception, e:
        logging.error(e, exc_info=True)

    # make a connection to master db
    try:
        conn = MySQLConnection(database=output_dbname, **config_info)
        cur = conn.cursor(buffered=True)
    except Exception, e:
        logging.error(e, exc_info=True)

    # create a table
    if coords == "mlt":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel float(9,2)," +\
                  " mag_glatc float(7,2)," +\
                  " mag_gltc float(8,2)," +\
                  " mag_gazmc TINYINT(5)," +\
                  " datetime DATETIME, " +\
                  " season VARCHAR(8), " +\
                  " rad VARCHAR(3), " +\
                  " CONSTRAINT all_rads PRIMARY KEY (" +\
                  "mag_glatc, mag_gltc, mag_gazmc, datetime, rad))"
    elif coords == "geo":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel float(9,2)," +\
                  " geo_glatc float(7,2)," +\
                  " geo_gltc float(8,2)," +\
                  " geo_gazmc TINYINT(5)," +\
                  " datetime DATETIME, " +\
                  " season VARCHAR(8), " +\
                  " rad VARCHAR(3), " +\
                  " CONSTRAINT all_rads PRIMARY KEY (" +\
                  "geo_glatc, geo_gltc, geo_gazmc, datetime, rad))"
    command = command.format(tb=output_table)
    try:
        cur.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)

    if coords == "mlt":
	command = "SELECT vel, mag_glatc, mag_gltc, mag_gazmc, " +\
		  "datetime, rad FROM {tb1} ORDER By datetime ASC"
    elif coords == "geo":
	command = "SELECT vel, geo_glatc, geo_gltc, geo_gazmc, " +\
		  "datetime, rad FROM {tb1} ORDER By datetime ASC"
    command = command.format(tb1=input_table)

    # check the db connection before fetching 
    if not conn_ten.is_connected():
	conn_ten.reconnect()
    # fetch the data
    try:
	cur_ten.execute(command)
    except Exception, e:
	logging.error(e, exc_info=True)
    rows = cur_ten.fetchall()

    # insert the data into a table
    if rows:
	if coords == "mlt":
	    command = "INSERT IGNORE INTO {tb2} (vel, mag_glatc, mag_gltc, " +\
		      "mag_gazmc, datetime, season, rad) VALUES (%s, %s, %s, %s, %s, %s, %s)"
	elif coords == "geo":
	    command = "INSERT IGNORE INTO {tb2} (vel, geo_glatc, geo_gltc, " +\
		      "geo_gazmc, datetime, season, rad) VALUES (%s, %s, %s, %s, %s, %s, %s)"
	command = command.format(tb2=output_table)
	for rw in rows:
	    vel, lat, lt, azm, dtm, rad = rw
            season = get_season_by_month(dtm.month)

	    # check the db connection before inserting
	    if not conn.is_connected():
		conn.reconnect()
	    # insert the data
	    try:
		cur.execute(command, (vel, lat, lt, azm, dtm, season, rad))
	    except Exception, e:
		logging.error(e, exc_info=True)

    # check the db connection before committing
    if not conn.is_connected():
	conn.reconnect()
    # commit the change
    try:
	conn.commit()
    except Exception, e:
	logging.error(e, exc_info=True)

    # close db connections
    conn_ten.close()
    conn.close()

    return

def master_summary(ftype="fitacf", baseLocation="../data/sqlite3/",
                   dbName=None):
    
    """ stores the summay of the master table into a different table in the same database.
    Time informatin is all lost at this point.
    """
    import sqlite3
    import datetime as dt
    import numpy as np 
    import logging

    # make db connection for dopsearch
    if dbName is None:
        dbName = "master_" + ftype + ".sqlite"

    # create a db that summarizes the data in the master table
    conn = sqlite3.connect(baseLocation + dbName)
    cur = conn.cursor()
    
    # create a summary table
    T1 = "master"
    T2 = "master_summary"
    cur.execute("CREATE TABLE IF NOT EXISTS {tb}\
                (vel TEXT, median_vel REAL, vel_count INTEGER, glatc REAL, glonc REAL, gazmc INTEGER,\
                 PRIMARY KEY (glatc, glonc, gazmc))".format(tb=T2))

    command = "INSERT OR IGNORE INTO {tb2} (vel, vel_count, glatc, glonc, gazmc)\
               SELECT group_concat(vel), COUNT(vel), glatc, glonc, gazmc FROM {tb1}\
               GROUP BY glatc, glonc, gazmc".format(tb1=T1, tb2=T2)
    cur.execute(command)

    # commit the change
    conn.commit()
    
    # select the velocity data grouping by lat-lon-azm bins
    command = "SELECT rowid, vel FROM {tb2}".format(tb2=T2)
    cur.execute(command)
    rws = cur.fetchall()

    for ii, rw in enumerate(rws):
        rwid, vel_txt = rw
        bin_vel = np.array([float(x) for x in vel_txt.split(",")])

        # take the median value
        median_vel = round(np.median(bin_vel),2)
        
        # populate the table 
        command = "UPDATE {tb} SET median_vel={median_vel}\
                  WHERE rowid=={rwid}".format(tb=T2, median_vel=median_vel, rwid=rwid)
        cur.execute(command)
        print ii

    # commit the change
    conn.commit()

    # close db connection
    conn.close()

    return

def main():
    import datetime as dt
    import logging

    # create a log file to which any error occured between client and
    # MySQL server communication will be written.
    logging.basicConfig(filename="./log_files/master_table_kp_00_to_23_hok_hkw.log",
                        level=logging.INFO)

    # input parameters
    input_table = "hok_hkw_kp_00_to_23_fitacf"
    output_table = "master_hok_hkw_kp_00_to_23"
    ftype = "fitacf"
    coords = "mlt"
    config_filename="../mysql_dbconfig_files/config.ini"
    section="midlat"
    input_dbname = "ten_min_median_" + coords + "_" + ftype
    output_dbname = "master_" + coords + "_" +ftype

    # build a master table 
    print "building a master table"
    build_master_table(input_table, output_table, ftype=ftype, coords=coords,
                       config_filename=config_filename,
                       section=section, input_dbname=input_dbname,
                       output_dbname=output_dbname)
    print "A master table is built"

#    # build a summary table
#    master_summary(ftype=ftype, baseLocation=baseLocation, dbName=None)
#    print "created the master_summary table"

    return

if __name__ == "__main__":
    main()

