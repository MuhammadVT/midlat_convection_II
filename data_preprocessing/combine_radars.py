def combine_radars(ftype="fitacf", coords="mlt",
                   config_filename="../mysql_dbconfig_files/config.ini",
                   section="midlat", db_name=None):
   
    """ combines all the ten-min median filtered gridded iscat data into
    one master table. Time informatin is all lost at this point.
    The results are stored in a different db file.

    Parameters
    ----------
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
        Name of the MySQL db to which the master grid data will be written.

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
    from mysql_dbutils import db_tools
    import logging

    # create db name
    if input_dbname is None:
        input_dbname = "ten_min_median_" + coords + "_" + ftype
    if output_dbname is None:
        output_dbname = "master_" + ftype
    input_table = "all_radars_" + ftype
    output_table = "master"

    # create a db (if not exist) for ten-min median data
    try:
        # create a db
        db_tools.create_db(output_dbname)
    except Exception, e:
        logging.error(e, exc_info=True)

    # read the db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make a connection to iscat db
    try:
        conn_iscat = MySQLConnection(database=iscat_dbname, **config_info)
        cur_iscat = conn_iscat.cursor(buffered=True)
    except Exception, e:
        logging.error(e, exc_info=True)






    # make db connection for dopsearch
    if dbName is None:
        dbName = "ten_min_median_" + ftype + ".sqlite"
    conn_ten = sqlite3.connect(baseLocation + dbName)
    cur_ten = conn_ten.cursor()

    # create a db that combines all the radar data into one master table
    conn = sqlite3.connect(baseLocation + "master_" + ftype + ".sqlite")
    cur = conn.cursor()

    # attach ten_min_median db into master db
    cur.execute("ATTACH DATABASE '{db}' as 'rads'".format(db=baseLocation+dbName))

    # create a table that combines all the radar data into one master table
    T1 = "master"
    cur.execute("CREATE TABLE IF NOT EXISTS {tb}\
                (vel REAL, glatc REAL, glonc REAL, gazmc INTEGER)"\
                .format(tb=T1))

    # get all the table names
    cur_ten.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    tbl_names = cur_ten.fetchall()
    tbl_names = [x[0] for x in tbl_names]

    for tbl in tbl_names:
        
        # copy data from ten_min_median db into master db
        command = "INSERT INTO {tb1} (vel, glatc, glonc, gazmc)\
                   SELECT vel, glatc, glonc, gazmc FROM {tb2}"\
                   .format(tb1=T1, tb2="rads."+tbl)
        cur.execute(command)

    # commit the change
    conn.commit()

    # detach ten_min_median db from master db
    cur.execute("DETACH DATABASE 'rads'")

    # close db connections
    conn_ten.close()
    conn.close()

    return

def master_summary(ftype="fitacf", baseLocation="../data/sqlite3/",
                   dbName=None):
    
    """ stores the summay of the master table into a different table in the same database.
    """
    import sqlite3
    import datetime as dt
    import numpy as np 

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

def worker(baseLocation):

    import datetime as dt

    # input parameters
    ftype = "fitacf"
    #ftype = "fitex"
            
    # combine the all the radars' data into one master table 
    combine_radars(ftype=ftype, baseLocation=baseLocation, dbName=None)
    print "created the master table"

    # build a summary table of the master table
    master_summary(ftype=ftype, baseLocation=baseLocation, dbName=None)
    print "created the master_summary table"


    return
if __name__ == "__main__":
    import multiprocessing

    seasons = ["winter", "summer", "equinox"]
    jobs = []
    for season in seasons:
        p = multiprocessing.Process(target=worker, args=(baseLocation,))
        jobs.append(p)
        p.start()

