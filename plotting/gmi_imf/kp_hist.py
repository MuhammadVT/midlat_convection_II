import matplotlib
matplotlib.use('Agg')

import sqlite3
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def plot_kp_hist(stm, etm, dbdir, dbname):

    """ Plots histograms of all kp for the period between stm and etm. 
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
    ax.set_title("Histogram of Kp between " + stm.strftime("%Y-%M-%d") +\
                 " and " +  etm.strftime("%Y-%M-%d"), fontsize="medium")
    ax.set_xlim([0, 9])
    ax.set_ylim([0, 6000])
    #ax.set_xticks(range(0, 10))
    #ax.set_xticks(range(10))
    #ax.set_xticks([0, 1.0, 1.3, 2.3, 4, 6, 9])
    #plt.xticks(rotation=90)

    return fig

def plot_previous_kp_hist(stm, etm, current_kp_lim, dbdir, dbname,
                          backward_shifted_hour=3):

    """ Plots histograms of kp before the current 3-hour intervals 
    for the period between stm and etm. 
    """

    # make a db connection
    conn = sqlite3.connect(dbdir + dbname, detect_types = sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    fig, ax = plt.subplots()
    
    input_table = "Kp"
    command = "SELECT kp FROM {tb} WHERE datetime IN " +\
              "(SELECT datetime(datetime, '-{backward_shifted_hour} hours') FROM {tb} " + \
              "WHERE (kp BETWEEN {kp_low} AND {kp_high}) AND " +\
              "(datetime BETWEEN '{stm}' AND '{etm}'))"
    command = command.format(tb=input_table,
                             backward_shifted_hour=backward_shifted_hour,
                             kp_low=current_kp_lim[0], kp_high=current_kp_lim[1],
                             stm=stm, etm=etm)
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
    ax.set_title("Histogram of Kp between " + stm.strftime("%Y-%M-%d") +\
                 " and " +  etm.strftime("%Y-%M-%d"), fontsize="medium")
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
    current_kp_lim = [0.0, 0.3]

    dbdir = "../../data/sqlite3/"
    
    # Plot the histogram of all the kp for the interval between stm and etm
    dbname = "gmi_imf.sqlite"
    fig = plot_kp_hist(stm, etm, dbdir, dbname)

    fig_path = "../plots/gmi_imf/"
    fig_name = "kp_" + stm.strftime("%Y%m%d") + "_" +\
               etm.strftime("%Y%m%d") + ".png"
    fig.savefig(fig_path+fig_name, dpi=300)

    # Plot the histogram of kp before the current 3-hour intervals 
    # for the period between stm and etm
    fig = plot_previous_kp_hist(stm, etm, current_kp_lim, dbdir, dbname,
                           backward_shifted_hour=3)

    fig_name = "previous_kp_hist_" +\
               "current_kp_" + str(current_kp_lim[0]) + "_to_" + str(current_kp_lim[1]) + "_" +\
               stm.strftime("%Y%m%d") + "_" +\
               etm.strftime("%Y%m%d") + ".png"
    fig.savefig(fig_path+fig_name, dpi=300)

