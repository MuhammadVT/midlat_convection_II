import pandas as pd
import numpy as np
import datetime as dt
import logging
import time
import aacgmv2
from dask.multiprocessing import get
from dask import dataframe as dd
from dask import delayed 
import multiprocessing as mp 

#def get_mlon(row):
#    # given the est bnd df, time get MLT from MLON
#    import aacgmv2
#    return np.round( aacgmv2.convert_mlt(row["mag_gltc"],
#                         row["datetime"], m2a=True), 1 )

def calc_mlon_slow(df):
    # given the est bnd df, time get MLT from MLON
    #df["mlon"] = df.apply(lambda x: np.round( aacgmv2.convert_mlt(x["mag_gltc"],
    #                      x["datetime"], m2a=True), 1), axis=1)
    mlon = df.apply(lambda x: np.round( aacgmv2.convert_mlt(x["mag_gltc"],
                          x["datetime"], m2a=True), 1), axis=1)
    return mlon

def calc_mlon(df):
    # given the est bnd df, time get MLT from MLON
    groups = df.groupby("datetime")
    mlon = np.zeros(df.shape[0])
    mlon.fill(np.nan)
    sidx = 0
    for name, g in groups:
        eidx = sidx + g.shape[0]
        mlon[sidx:eidx] = np.round(aacgmv2.convert_mlt(g["mag_gltc"].values,
                                pd.to_datetime(name), m2a=True), 1)
        sidx = eidx
    return mlon

def calc_eq_bnd_lat(df, df_poes):

    """
    Calculates Latitude value of the Equatorward Auroral Boundary
    """
    df = df.join(df_poes.set_index("datetime"), on="datetime",
                 how="left")
    mlat = df.p_0.values +  df.p_1.values * \
           np.cos(2*np.pi*(df.mlon.values/360.)+df.p_2.values)
    return mlat

def calc_eq_bnd_rel_lat(df):

    """
    Calculates the relative latitude value of the subauroral scatters
    """
    rel_mlat = np.round(df.mag_glatc.values - df.eq_aur_bnd_mlat.values)
    return rel_mlat

def worker(df, df_poes, mp_output=None):
    print("Calculating MLON...")
    #import pdb
    #pdb.set_trace()
    t0 = dt.datetime.now()
    #df["mlon"] = calc_mlon(df)
    df["mlon"] = calc_mlon(df)
    t1 = dt.datetime.now()
    print("Calculating MLON took "+str((t1-t0).total_seconds()/60.) + " minutes.")

    print("Calculating Equatorward Aur Bnd MLat...")
    df["eq_aur_bnd_mlat"] = calc_eq_bnd_lat(df, df_poes)

    print("Calculating Relative Equatorward Aur Bnd MLat...")
    df["rel_mlat"] = calc_eq_bnd_rel_lat(df)

    if mp_output is None:
        return df
    else:
        mp_output.put(df)
        return


def add_aur_bnd(sd_input_table, sd_output_table, poes_table,
                coords="mlt", ftype="fitacf",
                nbatches=20, db_name=None,
                config_filename="../mysql_dbconfig_files/config.ini",
                section="midlat"):

    """ Calculates Equatorward Bnd of Aur MLAT & rel_MLAT and writes 
    them to new columns in sd_input_table
    """

    from mysql.connector import MySQLConnection
    from sqlalchemy import create_engine
    import sys
    sys.path.append("../")
    from mysql_dbutils.db_config import db_config

    if db_name is None:
        db_name = "master_" + coords + "_" +ftype

    # read db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make a connection to ten-min median iscat db
    try:
        conn = MySQLConnection(database=db_name, **config_info)
        cur = conn.cursor(buffered=True)
    except Exception, e:
        logging.error(e, exc_info=True)

    # Fetch data from poes aur bnd table
    columns_poes = ["datetime", "p_0", "p_1", "p_2"]
    command_poes = "SELECT {columns_poes} FROM {poes_tb}"
    command_poes = command_poes.format(poes_tb=poes_table,
                                       columns_poes=",".join(columns_poes))

    tic = dt.datetime.now()
    print("Fetching data from POES Aur Bnd table. ")
    df_poes = fetch_data(conn,  command_poes, columns_poes)
    toc = dt.datetime.now()
    print("Fetching POES data took "+\
          str((toc-tic).total_seconds()/60.) + " minutes.\n")

    # Get the total number of rows
    command = "SELECT count(*) FROM {sd_tb}"
    command = command.format(sd_tb=sd_input_table)
    try:
        tic = dt.datetime.now()
        print("Calculating the # of rows in {sd_tb} table.".format(sd_tb=sd_input_table))
        cur.execute(command)
        nrows = cur.fetchall()[0][0]
        toc = dt.datetime.now()
        print("There are " + str(nrows) + " rows in " + sd_input_table + " table.")
        print("Calculating the # of rows took " + str((toc-tic).total_seconds()/60.) + " minutes.")
    except Exception, e:
        logging.error(e, exc_info=True)

    # Fetch data in parallel
    columns_sd = ["datetime", "vel", "mag_glatc", "mag_gltc",
                  "mag_gazmc", "rad"]
    batch_size = int(np.ceil(1.*nrows/nbatches))
    dfs = []
    mp_output = mp.Queue()
    procs = []
    for i in range(nbatches):
        offset = batch_size * i
        #command = "SELECT sd.datetime, sd.vel, sd.mag_glatc, " +\
        #          "sd.mag_gltc, sd.mag_gazmc, " +\
        #          "sd.rad, poes.p_0, poes.p_1, poes.p_2 " +\
        #          "FROM {sd_tb} AS sd LEFT JOIN {poes_tb} AS poes "+\
        #          "ON sd.datetime = poes.datetime LIMIT {batch_size} OFFSET {offset}"
        #command = command.format(sd_tb=sd_input_table, poes_tb=poes_table, 
        #                         batch_size=batch_size, offset=offset)
        command = "SELECT {columns_sd} FROM {sd_tb} LIMIT {batch_size} OFFSET {offset}"
        command = command.format(sd_tb=sd_input_table, batch_size=batch_size,
                                 offset=offset, columns_sd=",".join(columns_sd))

        print("\nStarting {i}th of {niters} iterations.".format(i=i+1, niters=nbatches))
        tic = dt.datetime.now()
        print("Fetching data from SD DB for {i}th iteration, ".format(i=i+1)+\
              "This will take some time...")
        df_i = fetch_data(conn,  command, columns_sd)
        toc = dt.datetime.now()
        print("Fetching data took "+\
              str((toc-tic).total_seconds()/60.) + " minutes.")

        # Calculate Equatorward Aur Bnd MLat & rel_MLat
        df_i = worker(df_i, df_poes) 
        toc1 = dt.datetime.now()
        print("{i}th iteration took ".format(i=i+1)+\
              str((toc1-tic).total_seconds()/60.) + " minutes.")
        dfs.append(df_i)

    print("Concatenating DFs...")
    df = pd.concat(dfs)
    df = df.reset_index().drop(["index"], axis=1)

    # Close DB connection
    try:
        conn.close()
    except Exception, e:
        logging.error(e, exc_info=True)


#    # Calculate Equatorward Aur Bnd MLat & rel_MLat
#    print("Converting Pandas DF to dask DF...")
#    ddf = dd.from_pandas(df, npartitions=nbatches)
#
#    print("Calculate MLON...")
#    ddf["mlon"] = ddf.map_partitions(lambda df: calc_mlon(df), meta=(None, 'f8')).compute()
#
#    print("Calculate Equatorward Aur Bnd MLat...")
#    ddf["eq_aur_bnd_mlat"] = ddf.map_partitions(lambda df: calc_eq_bnd_lat(df), meta=(None, 'f8')).compute()
#
#    print("Calculate Relative Equatorward Aur Bnd MLat...")
#    ddf["rel_mlat"] = ddf.map_partitions(lambda df: calc_eq_bnd_rel_lat(df), meta=(None, 'f8')).compute()
#
#    print("Joining Dask DF into one Pandas DF...")
#    df = ddf.compute()
#    df = df.reset_index().drop(["index"], axis=1)
#
#    try:
#        command ="ALTER TABLE {tb} ADD COLUMN mag_gmlonc float(7,2)".format(tb=sd_input_table) #        cur.execute(command) #    except:
#        # pass if the column mag_latc exists
#        pass
#    try:
#        command ="ALTER TABLE {tb} ADD COLUMN eq_bnd_mlat float(7,2)".format(tb=sd_input_table)
#        cur.execute(command)
#    except:
#        # pass if the column mag_ltc exists
#        pass
#    try:
#        command ="ALTER TABLE {tb} ADD COLUMN rel_mlat float(7,2)".format(tb=sd_input_table)
#        cur.execute(command)
#    except:
#        # pass if the column mag_ltc exists
#        pass
#

    print("Writing the output to {tb} table.".format(tb=sd_output_table))
    # Create a DB conn
    try:
        uri = "mysql://" + config_info["user"] + ":" +\
                           config_info["password"] + "@" +\
                           config_info["host"] +"/" +db_name
        conn = create_engine(uri)
    except Exception, e:
        logging.error(e, exc_info=True)
    # Write to DB
    df.to_sql(sd_output_table, conn, if_exists="replace",
              index=False)
    # Close DB connection
    try:
        conn.close()
    except Exception, e:
        logging.error(e, exc_info=True)

    return df

####################################
#    # Create a DB conn
#    try:
#        uri = "mysql://" + config_info["user"] + ":" +\
#                           config_info["password"] + "@" +\
#                           config_info["host"] +"/" +db_name
#        conn = create_engine(uri)
#    except Exception, e:
#        logging.error(e, exc_info=True)
#    try:
#        # Load data from DB to a Pandas DF
#        tic = dt.datetime.now()
#        print("Jointing two tables, this will take some time...")
#        #ddf = dd.read_sql_table(command, uri, index_col="datetime", npartitions=20)
#        #index_col = DATEDIFF(SECOND, '19000101', MyDateTimeColumn)
#        ddf = dd.read_sql_table(sd_input_table, uri, index_col="datetime", npartitions=20)
#        t0c = dt.datetime.now()
#        print("Jointing two tables took " + str((toc-tic).total_seconds()/60.) + " minutes")
#    except Exception, e:
#        print("Couldn't join two tables, returning None")
#        ddf = None
#        logging.error(e, exc_info=True)
#
#    #ddf = ddf.reset_index()
#    ddf["eq_bnd_lat"] = ddf.map_partitions(lambda df: calc_mlon(df), meta=(None, 'f8')).compute()
####################################



def fetch_data(conn, command, columns):
    try:
        cur = conn.cursor(buffered=True)
        cur.execute(command)
        rows = cur.fetchall()
        df = pd.DataFrame(data=rows, columns=columns)
    except Exception, e:
        df = None
        logging.error(e, exc_info=True)

    return df

def main():

    # input parameters
    rads_txt = "six_rads"
    #selected_years=[2011, 2012]
    #years_txt = "_years_" + "_".join([str(x) for x in selected_years])
    years_txt = ""
    kp_text = "_kp_00_to_23"
    #kp_text = "_kp_40_to_90"

    ftype = "fitacf"
    coords = "mlt"
    db_name = "ten_min_median_" + coords + "_" +ftype
    #sd_input_table = rads_txt + kp_text+ "_" +ftype
    sd_input_table = "test" 
    sd_output_table = "test_subauroral" 
    poes_table = "poes_aur_bnd_coeff"
    nbatches = 5

    # create a log file to which any error occured between client and
    # MySQL server communication will be written.
    logging.basicConfig(filename="./log_files/add_aur_bnd_" + rads_txt + \
                        kp_text + ".log", level=logging.INFO)

    df = add_aur_bnd(sd_input_table, sd_output_table, poes_table,
                     coords=coords, ftype=ftype,
                     nbatches=nbatches, db_name=db_name,
                     config_filename="../mysql_dbconfig_files/config.ini",
                     section="midlat")

    return df

if __name__ == "__main__":
    tic = dt.datetime.now()
    df = main()
    toc = dt.datetime.now()
    print("\nTotal time is " + str((toc-tic).total_seconds()/60.) + " minutes")

