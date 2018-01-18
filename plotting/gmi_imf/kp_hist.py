import matplotlib
matplotlib.use('Agg')

import sqlite3
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def plot_imf_hist(stm, etm, dbdir, dbname):

    """ Plots histograms of kp for the period between stm and etm. 
    """

    # make a db connection
    conn = sqlite3.connect(dbdir + dbname, detect_types = sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    fig, ax = plt.subplots()
    
    input_table = "Kp"
    command = "SELECT kp FROM {tb} " + \
              "WHERE datetime BETWEEN '{stm}' AND '{etm}' "
    command = command.format(tb=input_table, stm=stm, etm=etm)
    cur.execute(command)
    rws = cur.fetchall()
    kp = [x[0] for x in rws]

    # plot the histogram
    bns = [0, 0.31, 0.7, 1.31, 1.7, 2.31, 2.7, 3.31, 3.7, 4.31,
           4.7, 5.31, 5.7, 6.31, 6.7, 7.31, 7.7, 8.31, 8.7, 9] 
    ax.hist(kp, bins=bns, color="k", alpha=0.6)
    # set lables
    ax.set_xlabel("Kp")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Kp for " + stm.strftime("%Y-%M-%d") +\
                 " - " +  etm.strftime("%Y-%M-%d"), fontsize="medium")
    ax.set_xlim([0, 9])
    ax.set_ylim([0, 6000])
    #ax.set_xticks(range(0, 10))
    #ax.set_xticks(range(10))
    #ax.set_xticks([0, 1.0, 1.3, 2.3, 4, 6, 9])
    #plt.xticks(rotation=90)

    return fig

if __name__ == "__main__":

    import datetime as dt
    import numpy as np

    stm = dt.datetime(2011, 1, 1)
    etm = dt.datetime(2017, 1, 1)

    dbdir = "../../data/sqlite3/"
    
    # Plot the histogram of all the imf for the interval between stm and etm
    dbname = "gmi_imf.sqlite"
    fig = plot_imf_hist(stm, etm, dbdir, dbname)

    fig_path = "../plots/gmi_imf/"
    fig_name = "kp_" + stm.strftime("%Y%m%d") + "_" +\
               etm.strftime("%Y%m%d") + ".png"
    fig.savefig(fig_path+fig_name, dpi=300)


