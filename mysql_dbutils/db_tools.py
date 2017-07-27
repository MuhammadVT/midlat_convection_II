def create_db(db_name, config_filename="../mysql_dbconfig_files/config.ini",
              section="midlat"):
    """ Creates a MySQL table.

    Parameters
    ----------
    config_filename: str
	name and path of the configuration file
    section: str, default to mysql
	section of database configuration
  
    Returns
    -------
    Nothing

    """
    
    from mysql.connector import MySQLConnection
    from db_config import db_config

    # read the db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make db connection
    conn = MySQLConnection(**config_info)

    # create a db
    command = "CREATE DATABASE IF NOT EXISTS {db}".format(db=db_name) 
    conn.cursor().execute(command)

    # close the db connection
    conn.close()

    return 

