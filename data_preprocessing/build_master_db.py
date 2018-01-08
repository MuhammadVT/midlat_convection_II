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
                  " mag_gazmc SMALLINT," +\
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
                  " geo_gazmc SMALLINT," +\
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

def master_summary_by_season(input_table, output_table, coords="mlt", db_name=None,
                             config_filename="../mysql_dbconfig_files/config.ini",
                             section="midlat"):
    
    """ stores the summay statistics of the data in master table into 
    a different table in the same database.
    Time and rad informatin are all lost at this point.

    Parameters
    ----------
    input_table : str
        Name of a master table in master db
    output_table : str
        Name of a master_summary table in master db
    coords : str
        Coordinates in which the binning process took place.
        Default to "mlt, can be "geo" as well. 
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    db_name : str, default to None
        Name of the master db

    Returns
    -------
    Nothing

    """
    import datetime as dt
    import numpy as np 
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

    # create a table
    if coords == "mlt":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel_mean float(9,2)," +\
                  " vel_median float(9,2)," +\
                  " vel_std float(9,2)," +\
                  " vel_count INT," +\
                  " mag_glatc float(7,2)," +\
                  " mag_gltc float(8,2)," +\
                  " mag_gazmc SMALLINT," +\
                  " season VARCHAR(8), " +\
                  " CONSTRAINT grid_season PRIMARY KEY (" +\
                  "mag_glatc, mag_gltc, mag_gazmc, season))"
    elif coords == "geo":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel_mean float(9,2)," +\
                  " vel_median float(9,2)," +\
                  " vel_std float(9,2)," +\
                  " vel_count INT," +\
                  " geo_glatc float(7,2)," +\
                  " geo_gltc float(8,2)," +\
                  " geo_gazmc SMALLINT," +\
                  " season VARCHAR(8), " +\
                  " CONSTRAINT grid_season PRIMARY KEY (" +\
                  "geo_glatc, geo_gltc, geo_gazmc, season))"
    command = command.format(tb=output_table)
    try:
        cur.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)

    if coords == "mlt":
	command = "SELECT AVG(vel), STD(vel), COUNT(vel), " +\
                  "mag_glatc, mag_gltc, mag_gazmc, season " +\
		  "FROM {tb1} GROUP BY mag_glatc, mag_gltc, mag_gazmc, season"
    elif coords == "geo":
	command = "SELECT AVG(vel), STD(vel), COUNT(vel), " +\
                  "geo_glatc, geo_gltc, geo_gazmc, season " +\
		  "FROM {tb1} GROUP BY geo_glatc, geo_gltc, geo_gazmc, season"
    command = command.format(tb1=input_table)

    # check the db connection before fetching 
    if not conn.is_connected():
	conn.reconnect()
    # fetch the data
    try:
	cur.execute(command)
    except Exception, e:
	logging.error(e, exc_info=True)
    rows = cur.fetchall()

    # insert the data into a table
    if rows:
	if coords == "mlt":
	    command = "INSERT IGNORE INTO {tb2} (vel_mean, vel_median, vel_std, vel_count, " +\
                      "mag_glatc, mag_gltc, mag_gazmc, season) " +\
                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
	elif coords == "geo":
	    command = "INSERT IGNORE INTO {tb2} (vel_mean, vel_median, vel_std, vel_count, " +\
                      "geo_glatc, geo_gltc, geo_gazmc, season) " +\
                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
	command = command.format(tb2=output_table)
	for rw in rows:
            vel_mean, vel_std, vel_count, lat, lt, azm, season =rw

            # find median
            if coords == "mlt":
                command_tmp = "SELECT vel FROM {tb1} " +\
                              "WHERE mag_glatc={lat} and mag_gltc={lt} and "+\
                              "mag_gazmc={azm} and season='{season}'"
            elif coords == "geo":
                command_tmp = "SELECT vel FROM {tb1} " +\
                              "WHERE geo_glatc={lat} and geo_gltc={lt} and "+\
                              "geo_gazmc={azm} and season='{season}'"
            command_tmp = command_tmp.format(tb1=input_table, lat=lat, lt=lt,
                                             azm=azm, season=season)
            try:
                cur.execute(command_tmp)
            except Exception, e:
                logging.error(e, exc_info=True)
            vels_tmp = cur.fetchall()
            vels_tmp = [x[0] for x in vels_tmp]
            vel_median = np.median(vels_tmp)

	    # check the db connection before inserting
	    if not conn.is_connected():
		conn.reconnect()
	    # insert the data
	    try:
		cur.execute(command,
                            (round(vel_mean,2), round(vel_median,2), round(vel_std,2),
                             vel_count, lat, lt, azm, season))
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

def master_summary_by_radar_season(input_table, output_table, coords="mlt", db_name=None,
                                   radar_pair=["cvw", "cve"],
                                   config_filename="../mysql_dbconfig_files/config.ini",
                                   section="midlat"):
    
    """ stores the summay statistics of the data in master table into 
    a different table in the same database.
    Time and rad informatin are all lost at this point.
    NOTE: this function is only used selecet for pairs of radars from the
    six U.S. radars. 

    Parameters
    ----------
    input_table : str
        Name of a master table in master db
    output_table : str
        Name of a master_summary table in master db
    coords : str
        Coordinates in which the binning process took place.
        Default to "mlt, can be "geo" as well. 
    radar_pair = list
        a pair of three-character radar names
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    db_name : str, default to None
        Name of the master db

    Returns
    -------
    Nothing

    """
    import datetime as dt
    import numpy as np 
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

    # create a table
    if coords == "mlt":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel_mean float(9,2)," +\
                  " vel_median float(9,2)," +\
                  " vel_std float(9,2)," +\
                  " vel_count INT," +\
                  " mag_glatc float(7,2)," +\
                  " mag_gltc float(8,2)," +\
                  " mag_gazmc SMALLINT," +\
                  " season VARCHAR(8), " +\
                  " CONSTRAINT grid_season PRIMARY KEY (" +\
                  "mag_glatc, mag_gltc, mag_gazmc, season))"
    elif coords == "geo":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel_mean float(9,2)," +\
                  " vel_median float(9,2)," +\
                  " vel_std float(9,2)," +\
                  " vel_count INT," +\
                  " geo_glatc float(7,2)," +\
                  " geo_gltc float(8,2)," +\
                  " geo_gazmc SMALLINT," +\
                  " season VARCHAR(8), " +\
                  " CONSTRAINT grid_season PRIMARY KEY (" +\
                  "geo_glatc, geo_gltc, geo_gazmc, season))"

    command = command.format(tb=output_table)
    try:
        cur.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)

    if coords == "mlt":
	command = "SELECT AVG(vel), STD(vel), COUNT(vel), " +\
                  "mag_glatc, mag_gltc, mag_gazmc, season " +\
		  "FROM {tb1} "+\
                  "WHERE rad='{rad1}' OR rad='{rad2}' "+\
                  "GROUP BY mag_glatc, mag_gltc, mag_gazmc, season"
    elif coords == "geo":
	command = "SELECT AVG(vel), STD(vel), COUNT(vel), " +\
                  "geo_glatc, geo_gltc, geo_gazmc, season " +\
		  "FROM {tb1} "+\
                  "WHERE rad='{rad1}' OR rad='{rad2}' "+\
                  "GROUP BY geo_glatc, geo_gltc, geo_gazmc, season"
    command = command.format(tb1=input_table, rad1=radar_pair[0],
                             rad2=radar_pair[1])

    # check the db connection before fetching 
    if not conn.is_connected():
	conn.reconnect()
    # fetch the data
    try:
	cur.execute(command)
    except Exception, e:
	logging.error(e, exc_info=True)
    rows = cur.fetchall()

    # insert the data into a table
    if rows:
	if coords == "mlt":
	    command = "INSERT IGNORE INTO {tb2} (vel_mean, vel_median, vel_std, vel_count, " +\
                      "mag_glatc, mag_gltc, mag_gazmc, season) " +\
                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
	elif coords == "geo":
	    command = "INSERT IGNORE INTO {tb2} (vel_mean, vel_median, vel_std, vel_count, " +\
                      "geo_glatc, geo_gltc, geo_gazmc, season) " +\
                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
	command = command.format(tb2=output_table)
	for rw in rows:
            vel_mean, vel_std, vel_count, lat, lt, azm, season =rw

            # find median and std (if you want to recalculate std with setting max threshold to los velocities)
            if coords == "mlt":
                command_tmp = "SELECT vel FROM {tb1} " +\
                              "WHERE mag_glatc={lat} and mag_gltc={lt} and "+\
                              "mag_gazmc={azm} and season='{season}'"
            elif coords == "geo":
                command_tmp = "SELECT vel FROM {tb1} " +\
                              "WHERE geo_glatc={lat} and geo_gltc={lt} and "+\
                              "geo_gazmc={azm} and season='{season}'"
            command_tmp = command_tmp.format(tb1=input_table, lat=lat, lt=lt,
                                             azm=azm, season=season)
            try:
                cur.execute(command_tmp)
            except Exception, e:
                logging.error(e, exc_info=True)
            vels_tmp = cur.fetchall()
            vels_tmp = [x[0] for x in vels_tmp]
            vel_median = np.median(vels_tmp)
            vel_std = np.std([x for x in vels_tmp if np.abs(x) < 500.])

	    # check the db connection before inserting
	    if not conn.is_connected():
		conn.reconnect()
	    # insert the data
	    try:
		cur.execute(command,
                            (round(vel_mean,2), round(vel_median,2), round(vel_std,2),
                             vel_count, lat, lt, azm, season))
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

def master_summary_by_year_season(input_table, output_table, coords="mlt", db_name=None,
                                  selected_years=[2011, 2012],
                                  config_filename="../mysql_dbconfig_files/config.ini",
                                  section="midlat"):
    
    """ stores the summay statistics of the data in master table into 
    a different table in the same database.
    Time and rad informatin are all lost at this point.

    Parameters
    ----------
    input_table : str
        Name of a master table in master db
    output_table : str
        Name of a master_summary table in master db
    coords : str
        Coordinates in which the binning process took place.
        Default to "mlt, can be "geo" as well. 
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    db_name : str, default to None
        Name of the master db

    Returns
    -------
    Nothing

    """
    import datetime as dt
    import numpy as np 
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

    # create a table
    if coords == "mlt":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel_mean float(9,2)," +\
                  " vel_median float(9,2)," +\
                  " vel_std float(9,2)," +\
                  " vel_count INT," +\
                  " mag_glatc float(7,2)," +\
                  " mag_gltc float(8,2)," +\
                  " mag_gazmc SMALLINT," +\
                  " season VARCHAR(8), " +\
                  " CONSTRAINT grid_season PRIMARY KEY (" +\
                  "mag_glatc, mag_gltc, mag_gazmc, season))"
    elif coords == "geo":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel_mean float(9,2)," +\
                  " vel_median float(9,2)," +\
                  " vel_std float(9,2)," +\
                  " vel_count INT," +\
                  " geo_glatc float(7,2)," +\
                  " geo_gltc float(8,2)," +\
                  " geo_gazmc SMALLINT," +\
                  " season VARCHAR(8), " +\
                  " CONSTRAINT grid_season PRIMARY KEY (" +\
                  "geo_glatc, geo_gltc, geo_gazmc, season))"
    
    command = command.format(tb=output_table)
    try:
        cur.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)

    if coords == "mlt":
	command = "SELECT AVG(vel), STD(vel), COUNT(vel), " +\
                  "mag_glatc, mag_gltc, mag_gazmc, season " +\
		  "FROM {tb1} "+\
                  "WHERE YEAR(datetime) IN {years} "+\
                  "GROUP BY mag_glatc, mag_gltc, mag_gazmc, season"
    elif coords == "geo":
	command = "SELECT AVG(vel), STD(vel), COUNT(vel), " +\
                  "geo_glatc, geo_gltc, geo_gazmc, season " +\
		  "FROM {tb1} "+\
                  "WHERE YEAR(datetime) IN {years}"+\
                  "GROUP BY geo_glatc, geo_gltc, geo_gazmc, season"
    command = command.format(tb1=input_table, years=tuple(selected_years))
    print command

    # check the db connection before fetching 
    if not conn.is_connected():
	conn.reconnect()
    # fetch the data
    try:
	cur.execute(command)
    except Exception, e:
	logging.error(e, exc_info=True)
    rows = cur.fetchall()

    # insert the data into a table
    if rows:
	if coords == "mlt":
	    command = "INSERT IGNORE INTO {tb2} (vel_mean, vel_median, vel_std, vel_count, " +\
                      "mag_glatc, mag_gltc, mag_gazmc, season) " +\
                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
	elif coords == "geo":
	    command = "INSERT IGNORE INTO {tb2} (vel_mean, vel_median, vel_std, vel_count, " +\
                      "geo_glatc, geo_gltc, geo_gazmc, season) " +\
                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
	command = command.format(tb2=output_table)
	for rw in rows:
            vel_mean, vel_std, vel_count, lat, lt, azm, season =rw

            # find median and std
            if coords == "mlt":
                command_tmp = "SELECT vel FROM {tb1} " +\
                              "WHERE mag_glatc={lat} and mag_gltc={lt} and "+\
                              "mag_gazmc={azm} and season='{season}' and " +\
                              "YEAR(datetime) IN {years}"
            elif coords == "geo":
                command_tmp = "SELECT vel FROM {tb1} " +\
                              "WHERE geo_glatc={lat} and geo_gltc={lt} and "+\
                              "geo_gazmc={azm} and season='{season}' and " +\
                              "YEAR(datetime) IN {years}"
            command_tmp = command_tmp.format(tb1=input_table, lat=lat, lt=lt,
                                             azm=azm, season=season,
                                             years=tuple(selected_years))
            try:
                cur.execute(command_tmp)
            except Exception, e:
                logging.error(e, exc_info=True)
            vels_tmp = cur.fetchall()
            vels_tmp = [x[0] for x in vels_tmp]
            vel_median = np.median(vels_tmp)
            vel_std = np.std([x for x in vels_tmp if np.abs(x) < 500.])
            if np.isnan(vel_std):
                vel_std = np.std(vels_tmp)

	    # check the db connection before inserting
	    if not conn.is_connected():
		conn.reconnect()
	    # insert the data
	    try:
		cur.execute(command,
                            (round(vel_mean,2), round(vel_median,2), round(vel_std,2),
                             vel_count, lat, lt, azm, season))
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

def master_summary_by_month(input_table, output_table, coords="mlt", db_name=None,
                            config_filename="../mysql_dbconfig_files/config.ini",
                            section="midlat", pseudo_month=False):
    
    """ stores the summay statistics of the data in master table into 
    a different table in the same database.
    Time and rad informatin are all lost at this point.

    Parameters
    ----------
    input_table : str
        Name of a master table in master db
    output_table : str
        Name of a master_summary table in master db
    coords : str
        Coordinates in which the binning process took place.
        Default to "mlt, can be "geo" as well. 
    radar_pair = list
        a pair of three-character radar names
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    db_name : str, default to None
        Name of the master db

    Returns
    -------
    Nothing

    """
    import datetime as dt
    import numpy as np 
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

    # create a table
    if coords == "mlt":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel_mean float(9,2)," +\
                  " vel_median float(9,2)," +\
                  " vel_std float(9,2)," +\
                  " vel_count INT," +\
                  " mag_glatc float(7,2)," +\
                  " mag_gltc float(8,2)," +\
                  " mag_gazmc SMALLINT," +\
                  " month SMALLINT," +\
                  " CONSTRAINT grid_month PRIMARY KEY (" +\
                  "mag_glatc, mag_gltc, mag_gazmc, month))"
    elif coords == "geo":
        command = "CREATE TABLE IF NOT EXISTS {tb}" +\
                  "(vel_mean float(9,2)," +\
                  " vel_median float(9,2)," +\
                  " vel_std float(9,2)," +\
                  " vel_count INT," +\
                  " geo_glatc float(7,2)," +\
                  " geo_gltc float(8,2)," +\
                  " geo_gazmc SMALLINT," +\
                  " month SMALLINT," +\
                  " CONSTRAINT grid_month PRIMARY KEY (" +\
                  "geo_glatc, geo_gltc, geo_gazmc, month))"

    command = command.format(tb=output_table)
    try:
        cur.execute(command)
    except Exception, e:
        logging.error(e, exc_info=True)

    if pseudo_month:
        if coords == "mlt":
            command = "SELECT AVG(vel), STD(vel), COUNT(vel), " +\
                      "mag_glatc, mag_gltc, mag_gazmc, MONTH(ADDDATE(datetime, INTERVAL -6 day)) " +\
                      "FROM {tb1} "+\
                      "GROUP BY mag_glatc, mag_gltc, mag_gazmc, MONTH(ADDDATE(datetime, INTERVAL -6 day))"
        elif coords == "geo":
            command = "SELECT AVG(vel), STD(vel), COUNT(vel), " +\
                      "geo_glatc, geo_gltc, geo_gazmc, MONTH(ADDDATE(datetime, INTERVAL -6 day)) " +\
                      "FROM {tb1} "+\
                      "GROUP BY geo_glatc, geo_gltc, geo_gazmc, MONTH(ADDDATE(datetime, INTERVAL -6 day))"
    else:
        if coords == "mlt":
            command = "SELECT AVG(vel), STD(vel), COUNT(vel), " +\
                      "mag_glatc, mag_gltc, mag_gazmc, MONTH(datetime) " +\
                      "FROM {tb1} "+\
                      "GROUP BY mag_glatc, mag_gltc, mag_gazmc, MONTH(datetime)"
        elif coords == "geo":
            command = "SELECT AVG(vel), STD(vel), COUNT(vel), " +\
                      "geo_glatc, geo_gltc, geo_gazmc, MONTH(datetime) " +\
                      "FROM {tb1} "+\
                      "GROUP BY geo_glatc, geo_gltc, geo_gazmc, MONTH(datetime)"

    command = command.format(tb1=input_table)
    print command

    # check the db connection before fetching 
    if not conn.is_connected():
	conn.reconnect()
    # fetch the data
    try:
	cur.execute(command)
    except Exception, e:
	logging.error(e, exc_info=True)
    rows = cur.fetchall()

    # insert the data into a table
    if rows:
	if coords == "mlt":
	    command = "INSERT IGNORE INTO {tb2} (vel_mean, vel_median, vel_std, vel_count, " +\
                      "mag_glatc, mag_gltc, mag_gazmc, month) " +\
                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
	elif coords == "geo":
	    command = "INSERT IGNORE INTO {tb2} (vel_mean, vel_median, vel_std, vel_count, " +\
                      "geo_glatc, geo_gltc, geo_gazmc, month) " +\
                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
	command = command.format(tb2=output_table)
	for rw in rows:
            vel_mean, vel_std, vel_count, lat, lt, azm, month =rw

            # find median
            if pseudo_month:
                if coords == "mlt":
                    command_tmp = "SELECT vel FROM {tb1} " +\
                                  "WHERE mag_glatc={lat} and mag_gltc={lt} and "+\
                                  "mag_gazmc={azm} and MONTH(ADDDATE(datetime, INTERVAL -6 day))={month}"
                elif coords == "geo":
                    command_tmp = "SELECT vel FROM {tb1} " +\
                                  "WHERE geo_glatc={lat} and geo_gltc={lt} and "+\
                                  "geo_gazmc={azm} and MONTH(ADDDATE(datetime, INTERVAL -6 day))={month}"
            else:
                if coords == "mlt":
                    command_tmp = "SELECT vel FROM {tb1} " +\
                                  "WHERE mag_glatc={lat} and mag_gltc={lt} and "+\
                                  "mag_gazmc={azm} and MONTH(datetime)={month}"
                elif coords == "geo":
                    command_tmp = "SELECT vel FROM {tb1} " +\
                                  "WHERE geo_glatc={lat} and geo_gltc={lt} and "+\
                                  "geo_gazmc={azm} and MONTH(datetime)={month}"
            command_tmp = command_tmp.format(tb1=input_table, lat=lat, lt=lt,
                                             azm=azm, month=month)
            try:
                cur.execute(command_tmp)
            except Exception, e:
                logging.error(e, exc_info=True)
            vels_tmp = cur.fetchall()
            vels_tmp = [x[0] for x in vels_tmp]
            vel_median = np.median(vels_tmp)

	    # check the db connection before inserting
	    if not conn.is_connected():
		conn.reconnect()
	    # insert the data
	    try:
		cur.execute(command,
                            (round(vel_mean,2), round(vel_median,2), round(vel_std,2),
                             vel_count, lat, lt, azm, month))
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



def main(master_table=True, master_summary_table=True):
    import datetime as dt
    import logging

    # create a log file to which any error occured between client and
    # MySQL server communication will be written.
    logging.basicConfig(filename="./log_files/master_table_kp_00_to_23_six_rads_2011_2012.log",
                        level=logging.INFO)

    # input parameters
    #input_table_1 = "hok_hkw_kp_00_to_23_fitacf"
    #output_table_1 = "master_hok_hkw_kp_00_to_23"
    #input_table_2 = "master_hok_hkw_kp_00_to_23"
    #output_table_2 = "master_summary_hok_hkw_kp_00_to_23"

    #input_table_1 = "ade_adw_kp_00_to_23_fitacf"
    #output_table_1 = "master_ade_adw_kp_00_to_23"
    #input_table_2 = "master_ade_adw_kp_00_to_23"
    #output_table_2 = "master_summary_ade_adw_kp_00_to_23"


    #rads_txt = "bks_wal"
    rads_txt = "six_rads"
    #rads_txt = "ade_adw"

    selected_years=[2011, 2012]
    years_txt = "_years_" + "_".join([str(x) for x in selected_years])
    #years_txt = ""

    input_table_1 = rads_txt + "_kp_00_to_23_fitacf"
    output_table_1 = "master_" + rads_txt + "_kp_00_to_23"
    input_table_2 = "master_" + rads_txt + "_kp_00_to_23"
    output_table_2 = "master_summary_" + rads_txt + "_kp_00_to_23" + years_txt
    #output_table_2 = "master_summary_" + rads_txt + "_kp_00_to_23_by_pseudo_month"

    ftype = "fitacf"
    coords = "mlt"
    config_filename="../mysql_dbconfig_files/config.ini"
    section="midlat"
    input_dbname = "ten_min_median_" + coords + "_" + ftype
    output_dbname = "master_" + coords + "_" +ftype

    if master_table:
        # build a master table 
        print "building a master table"
        build_master_table(input_table_1, output_table_1, ftype=ftype, coords=coords,
                           config_filename=config_filename,
                           section=section, input_dbname=input_dbname,
                           output_dbname=output_dbname)
        print "A master table has been built"

    if master_summary_table:
        # build a summary table
        print "building a master_summary table"

#        master_summary_by_month(input_table_2, output_table_2, coords=coords,
#                                db_name=output_dbname,
#                                config_filename="../mysql_dbconfig_files/config.ini",
#                                section="midlat", pseudo_month=False)

#        master_summary_by_season(input_table_2, output_table_2, coords=coords,
#                               db_name=output_dbname,
#                               config_filename="../mysql_dbconfig_files/config.ini",
#                               section="midlat")

#        master_summary_by_radar_season(input_table_2, output_table_2, coords=coords,
#                                       db_name=output_dbname,
#                                       radar_pair=["cve", "cvw"],
#                                       config_filename="../mysql_dbconfig_files/config.ini",
#                                       section="midlat")

        master_summary_by_year_season(input_table_2, output_table_2, coords=coords,
                                       db_name=output_dbname,
                                       selected_years=selected_years,
                                       config_filename="../mysql_dbconfig_files/config.ini",
                                       section="midlat")

        print "A master_summary has been build"

    return

if __name__ == "__main__":
    main(master_table=False, master_summary_table=True)

