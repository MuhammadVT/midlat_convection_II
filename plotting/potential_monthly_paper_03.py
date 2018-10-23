import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import datetime as dt
import pandas as pd
import numpy as np

def fetch_data(input_table, lat_range=[52, 59],
               lt_range=[344, 346], 
               nvel_min=250,
               config_filename="../mysql_dbconfig_files/config.ini",
               section="midlat", db_name=None, ftype="fitacf",
	       coords="mlt", sqrt_weighting=True, limit_to_night=True):

    """ fetch fitted data from the master db into a dict

    Parameters
    ----------
    input_table : str
        A table name in db_name db
    lat_ragne : list
	The range of latitudes of interest
    nvel_min : int
        minimum requirement for the number of velocity measurements
	in a lat-lon grid cell
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
    sqrt_weighting : bool
        if set to True, the fitted vectors that are produced through weighting the
        number of points within each azimuthal bin will be retrieved. 
        if set to False, the fitted vectors that are produced by equality weighting
        the number of points within each azimuthal bin will be retrieved.

    Return
    ------
    data_dict : dict

    """
    import sqlite3
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

    # set input_table name
    if sqrt_weighting:
        input_table = input_table
    else:
        input_table = input_table + "_equal_weighting"

    # formulate column names
    col_glatc = "mag_glatc"   # glatc -> gridded latitude center
    col_gltc = "mag_gltc"     # mlt hour in degrees
    col_gazmc = "mag_gazmc"   # gazmc -> gridded azimuthal center
    col_gazmc_count = "mag_gazmc_count"
    command = "SELECT vel_count, vel_mag, vel_dir, {glatc}, {gltc}, " +\
              "vel_mag_err, vel_dir_err, month FROM {tb1} " +\
              "WHERE ({glatc} BETWEEN {lat_min} AND {lat_max}) " +\
              "AND ({gltc} BETWEEN {lt_min} AND {lt_max}) "

    command = command.format(tb1=input_table, glatc=col_glatc,
                             gltc = col_gltc, lat_min=lat_range[0],
                             lat_max=lat_range[1], lt_min=lt_range[0],
                             lt_max=lt_range[1])
    cur.execute(command)
    rws = cur.fetchall()

    # close db connection
    conn.close()

    data_dict = {}
    # filter the data based on lattitude range. 
    data_dict['vel_count'] = np.array([x[0] for x in rws if x[0] >= nvel_min])
    data_dict['vel_mag'] = np.array([x[1] for x in rws if x[0] >= nvel_min])
    data_dict['vel_dir'] = np.array([x[2] for x in rws if x[0] >= nvel_min])
    data_dict['glatc'] = np.array([x[3] for x in rws if x[0] >= nvel_min])
    data_dict['glonc'] = np.array([x[4] for x in rws if x[0] >= nvel_min])
    data_dict['vel_mag_err'] = np.array([x[5] for x in rws if x[0] >= nvel_min])
    data_dict['vel_dir_err'] = np.array([x[6] for x in rws if x[0] >= nvel_min])
    data_dict['month'] = np.array([x[7] for x in rws])

    # calculate velocity components
    vel_mag = data_dict['vel_mag']
    vel_dir = np.deg2rad(data_dict['vel_dir'])
    data_dict['vel_mag_zonal'] = vel_mag*(-1.0)*np.sin(vel_dir)
    data_dict['vel_mag_meridional'] = vel_mag*(-1.0)*np.cos(vel_dir)

    # Calculate E-fields
    lat_to_Bvertical = {52.5 : (42 + 0.6*0)  * 1e3,
                        53.5 : (42 + 0.6*1)  * 1e3,
                        54.5 : (42 + 0.6*2)  * 1e3,
                        55.5 : (42 + 0.6*3)  * 1e3,
                        56.5 : (42 + 0.6*4)  * 1e3,
                        57.5 : (42 + 0.6*5)  * 1e3,
                        58.5 : (42 + 0.6*6)  * 1e3,
                        59.5 : (42 + 0.6*7)  * 1e3,
                        60.5 : (42 + 0.6*8)  * 1e3,
                        61.5 : (42 + 0.6*9)  * 1e3,
                        62.5 : (42 + 0.6*10) * 1e3}  # nT
    Bvertical = np.array([lat_to_Bvertical[x] for x in data_dict["glatc"]])
    data_dict['E-poleward'] = (-1.) * np.multiply(data_dict["vel_mag_zonal"], Bvertical) * 1e-9  # V/m
    del_dist = 110 * 1e3    # meter
    data_dict['Phi-poleward'] = data_dict['E-poleward'] * del_dist    # V

    return data_dict

def main():

    # input parameters
    nvel_min=100
    lat_range=[52, 60]
    lt_range=[344, 346]     # MLT in degrees. Divide by 15. to convert to hour 

    subauroral_only=True
    if subauroral_only:
        subaur_text = "_subaur"
    else:
        subaur_text = ""

    ftype = "fitacf"
    coords = "mlt"
    sqrt_weighting = True
    rads_txt = "six_rads"

    month_txt = "by_month"
    input_table = "master_cosfit_" + rads_txt + "_kp_00_to_23_" + month_txt + subaur_text

    # fetches the data from db 
    data_dict = fetch_data(input_table, lat_range=lat_range,
                lt_range=lt_range, nvel_min=nvel_min,
                config_filename="../mysql_dbconfig_files/config.ini",
                section="midlat", db_name=None, ftype=ftype,
                coords=coords, sqrt_weighting=sqrt_weighting)

    df = pd.DataFrame(data=data_dict)

    # Calculate Potential difference between lat-range
    pot = df[["month", "Phi-poleward"]].groupby("month").sum().round().to_dict()
    pot = pot["Phi-poleward"]

    fig, ax = plt.subplots()
    ax.plot(pot.keys(), pot.values())
    ax.set_xlabel("Month")
    ax.set_ylabel("Potential [Volt]")
    fig.savefig("./potential_vs_month.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return pot

if __name__ == "__main__":
    pot = main()

