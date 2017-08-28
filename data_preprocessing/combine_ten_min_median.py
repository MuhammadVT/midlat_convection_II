def combine_ten_min_median(rads, ftype="fitacf", coords="mlt",
                           config_filename="../mysql_dbconfig_files/config.ini",
                           section="midlat", db_name=None):

    """ combines ten-minute median filtered gridded data from radars 
    specified by rads argument into a single table.

    Parameters
    ----------
    rads : list
        A list of three-letter radar codes
    ftype : str
        SuperDARN file type
    coords : str
        The Coordinate system. Default to "mlt. Can be "geo" as well.
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    db_name : str, default to None
        Name of the MySQL db where ten-min median data is stored.

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

    # consruct a db name and a table name
    if db_name is None:
        db_name = "ten_min_median_" + coords + "_" + ftype
    output_table = "all_radars_" + ftype

    # read the db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make a connection to ten-min median db
    try:
        conn = MySQLConnection(database=db_name, **config_info)
        cur = conn.cursor(buffered=True)
    except Exception, e:
        logging.error(e, exc_info=True)

    # create a table that combines gridded data from all the radar
    if coords == "mlt":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel float(9,2)," +\
                  " mag_glatc float(7,2)," +\
                  " mag_gltc float(8,2)," +\
                  " mag_gazmc TINYINT(5)," +\
                  " datetime DATETIME, " +\
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
                  " rad VARCHAR(3), " +\
                  " CONSTRAINT all_rads PRIMARY KEY (" +\
                  "geo_glatc, geo_gltc, geo_gazmc, datetime, rad))"
    command = command.format(tb=output_table)
    try:
        cur.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)

    # construct table names in iscat db
    tbl_names = [rad + "_" + ftype for rad in rads]

    # move the data between tables 
    for i, tbl_name in enumerate(tbl_names):
        rad = tbl_name.split("_")[0]
        if coords == "mlt":
            command = "SELECT vel, mag_glatc, mag_gltc, mag_gazmc, " +\
                      "datetime FROM {tb1} ORDER By datetime ASC"
        elif coords == "geo":
            command = "SELECT vel, geo_glatc, geo_gltc, geo_gazmc, " +\
                      "datetime FROM {tb1} ORDER By datetime ASC"
        command = command.format(tb1=tbl_name)

        # check the db connection before fetching 
        if not conn.is_connected():
            conn.reconnect()
        # fetch the data
        try:
            cur.execute(command)
        except Exception, e:
            logging.error(e, exc_info=True)
        rows = cur.fetchall()

        # run the data into a table
        if rows:
            if coords == "mlt":
                command = "INSERT IGNORE INTO {tb2} (vel, mag_glatc, mag_gltc, " +\
                          "mag_gazmc, datetime, rad) VALUES (%s, %s, %s, %s, %s, %s)"
            elif coords == "geo":
                command = "INSERT IGNORE INTO {tb2} (vel, geo_glatc, geo_gltc, " +\
                          "geo_gazmc, datetime, rad) VALUES (%s, %s, %s, %s, %s, %s)"
            command = command.format(tb2=output_table)
            for rw in rows:
                vel, lat, lt, azm, dtm = rw

		# check the db connection before inserting
		if not conn.is_connected():
		    conn.reconnect()
		# insert the data
                try:
                    cur.execute(command, (vel, lat, lt, azm, dtm, rad))
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
    conn.close()

    return

def main():
    """executes the above function."""

    import datetime as dt
    import logging


    # create a log file to which any error occured between client and
    # MySQL server communication will be written.
    logging.basicConfig(filename="./log_files/combine_ten_min_median_japan.log",
                        level=logging.INFO)
 
    # input parameters
    rads = ["hok", "hkw"] 
    ftype="fitacf"
    coords="mlt"
    config_filename="../mysql_dbconfig_files/config.ini"
    section="midlat"
    db_name=None

    # take ten minutes median values
    print("moving ten_min_median of " + str(rads) + " into all_radars_" +\
	  ftype + " table")
    t1 = dt.datetime.now()
    combine_ten_min_median(rads, ftype=ftype, coords=coords,
                           config_filename=config_filename,
                           section=section, db_name=db_name)
    t2 = dt.datetime.now()
    print("Finished moving. It took " + str((t2-t1).total_seconds() / 60.) + " mins\n")

    return

if __name__ == "__main__":
    main()

