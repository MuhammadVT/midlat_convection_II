#import matplotlib
#matplotlib.use('Agg')

import sqlite3
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def find_imf_turning(stm, etm, dbdir, dbname, table_1, table_2, max_turn_time=10.*60):
def find_imf_turning(stm, etm, dbdir, dbname, table_1, table_2,
                     max_turn_time=10.*60):

    """ 
    """

    import numpy as np
    import pandas as pd

    # make a db connection
    conn = sqlite3.connect(dbdir + dbname, detect_types = sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()
    
    command = "SELECT Bx, By, Bz, clock_angle, datetime FROM " +\
	      "(SELECT Bx, By, Bz, clock_angle, datetime FROM {tb1} " + \
              "UNION ALL " +\
              "SELECT Bx, By, Bz, clock_angle, datetime FROM {tb2}) as tb " + \
              "WHERE datetime BETWEEN '{stm}' AND '{etm}' ORDER BY DATETIME ASC "
    command = command.format(tb1=table_1, tb2=table_2, stm=stm, etm=etm)
    cur.execute(command)
    rws = cur.fetchall()
    Bx = [x[0] for x in rws]
    By = [x[1] for x in rws]
    Bz = [x[2] for x in rws]
    theta = [x[3] for x in rws]
    dtm = [x[4] for x in rws]
    df = pd.DataFrame(data={"Bx":Bx, "By":By, "Bz":Bz, "theta":theta},
                      index=dtm)

    # Add turning time
    dt_vals = df.shift(1).index[1:] - df.index[:-1]
    #dt_vals = [np.nan] + [round(x.total_seconds() / 60., 2) for x in dt_vals]  # in minutes
    dt_vals = [np.nan] + [round(x.total_seconds(), 2) for x in dt_vals]  # in seconds 
    df.loc[:, "turning_time"] = dt_vals 

    # Find turning points
    dtm_turn = df.

    return df

if __name__ == "__main__":

    import datetime as dt
    import numpy as np

    stm = dt.datetime(2011, 1, 1)
    etm = dt.datetime(2011, 1, 4)

    dbdir = "../../data/sqlite3/" 
    dbname = "binned_imf.sqlite"

    # bin IMF clock angle
    # construct input table names
    bvec_max = 0.95
    before_mins=20
    after_mins=0
    del_tm=10
    bns = [[150, 210], [-30, 30]]
    tbls = []
    for bn in bns:
	tbl = "b" + str((bn[0]%360)) + "_b" + str(bn[1]%360) + \
	      "_before" + str(before_mins) + "_after" +  str(after_mins) + \
	      "_bvec" + str(bvec_max).split('.')[-1] + "_all"
	tbls.append(tbl)
    table_1, table_2 = tbls

    df = find_imf_turning(stm, etm, dbdir, dbname, table_1, table_2)
