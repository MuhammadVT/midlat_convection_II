import datetime as dt
import numpy as np
import dask
import logging

def read_fitcoeff_to_ddf(stime, etime, file_dir="../../data/poes/bnd_fitcoeff/",
                         file_format="txt"):
    """
    Reads POES Aur Bnd Fitting Coefficients into a Dusk DF.
    """
    import dask.dataframe as dd

    files = file_dir + "*" +  file_format

    # Read files in parallel
    print "reading boundary data from-->", file_dir
    colTypeDict = { "p_0" : np.float64, "p_1" : np.float64,
                    "p_2" : np.float64, "sat":np.str,
                    "date":np.str, "time":np.str }
    date_parser = lambda date, time: dt.datetime.strptime(date+time, "%Y%m%d%H%M")
    ddf = dd.read_csv(files, delim_whitespace=True,
                      parse_dates={"datetime":[4, 5]},
		      date_parser=date_parser,
                      dtype=colTypeDict)

    # Select data between stime and etime
    ddf = ddf.loc[(ddf.datetime >= stm) & (ddf.datetime <= etm), :]
    print("Loaded files into Dusk DataFrame")

    return ddf

def move_to_db(ddf, table_name, ftype="fitacf", coords="mlt",
	       db_name=None, 
               config_filename="../../mysql_dbconfig_files/config.ini",
               section="midlat"):

    """Writes POES Aur Bnd fitted coeffs to MySQL DB"""

    from mysql.connector import MySQLConnection
    from sqlalchemy import create_engine
    import sys
    sys.path.append("../../")
    from mysql_dbutils.db_config import db_config

    # construct a db name
    if db_name is None:
        db_name = "master_" + coords + "_" +ftype
 
    # read db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make a connection to master db
    try:
        engine = create_engine("mysql://" + config_info["user"] + ":" +\
			       config_info["password"] + "@" +\
			       config_info["host"] + "/" + db_name)
        conn = engine.connect() 
    except Exception, e:
        logging.error(e, exc_info=True)
 
    # create a table
    command = "CREATE TABLE IF NOT EXISTS {tb}" +\
	      "(datetime DATETIME," +\
	      " p_0 float(5,2) DEFAULT NULL," +\
	      " p_1 float(5,2) DEFAULT NULL," +\
	      " p_2 float(5,2) DEFAULT NULL," +\
	      " sat TEXT DEFAULT NULL," +\
	      " CONSTRAINT fit_coeff PRIMARY KEY (datetime))"
    command = command.format(tb=table_name)

    # Convert Dusk DF to Pandas DF
    df = ddf.compute()
    print("Convered Dusk DF to Pandas Df")

    # Set the datetime to HH:M5
    df.datetime = df.datetime.apply(lambda x: x + dt.timedelta(minutes=5))

    # Move pandas DF to db
    try:
	df.to_sql(table_name, conn, schema=None, if_exists='append',
		  index=False, index_label=None, chunksize=10000, dtype=None)
    except Exception, e:
        logging.error(e, exc_info=True)

    # Close DB connection
    try:
	conn.close()
    except Exception, e:
        logging.error(e, exc_info=True)

    return

if __name__ == "__main__":

    stm = dt.datetime( 2011, 1, 1 )
    etm = dt.datetime( 2018, 7, 1 )
    file_dir = "../../data/poes/bnd_fitcoeff/"
    table_name = "poes_aur_bnd_coeff"
    db_name = None

    # create a log file to which any error occured between client and
    # MySQL server communication will be written.
    logging.basicConfig(filename="../log_files/poes_aur_bnd_to_db.log",
                        level=logging.INFO)

    # Read POES Aur Bnd Fitting Coeffs to Dusk DF
    ddf = read_fitcoeff_to_ddf(stm, etm, file_dir=file_dir,
			       file_format="txt")

    # Store data to DB
    move_to_db(ddf, table_name, ftype="fitacf", coords="mlt",
	       db_name=db_name, 
               config_filename="../../mysql_dbconfig_files/config.ini",
               section="midlat")

