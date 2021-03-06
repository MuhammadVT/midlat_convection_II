import matplotlib
matplotlib.use('Agg')
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def plot_imf_hist(stm, etm, dbdir, dbname, ax=None, param="theta"):

    """ Plots histograms of 1-min imf from omni for the period between stm and etm. 
    NOTE: No stability condition is inforced here
    """

    # make a db connection
    conn = sqlite3.connect(dbdir + dbname, detect_types = sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    if ax is None:
        fig, ax = plt.subplots()
    
    input_table = "IMF"
    command = "SELECT Bx, By, Bz, theta FROM {tb} " + \
              "WHERE datetime BETWEEN '{stm}' AND '{etm}' "
    command = command.format(tb=input_table, stm=stm, etm=etm)
    cur.execute(command)
    rws = cur.fetchall()
    Bx = [x[0] for x in rws]
    By = [x[1] for x in rws]
    Bz = [x[2] for x in rws]
    theta = [x[3] for x in rws]

    # plot the histogram
    if param == "theta":
        ax.hist(theta, bins=(360+0)/5-1, range=(0, 360), color="k", alpha=0.6)
        # set lables
        ax.set_xlabel("IMF Clock Angle [degree]")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of IMF Clock Angle", fontsize="medium")
        ax.set_xlim([0, 360])
        ax.set_ylim([0, 55000])
        ax.set_xticks(range(0, 450, 90))

    if param == "Bx":
        var = Bx
    if param == "By":
        var = By
    if param == "Bz":
        var = Bz
    ax.hist(var, bins=61, range=(-30, 30), color="k", alpha=0.6)
    # set lables
    ax.set_xlabel(param + " [nT]")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of " + param, fontsize="medium")
    ax.set_xlim([-30, 30])
    ax.set_ylim([0, 450000])


    return

def plot_binned_imf_clock_angle_hist(stm, etm, dbdir, dbname, ax=None, bvec_max=0.95, before_mins=20, 
			      after_mins=10, del_tm=10, sector_center_dist = 90,
			      bins=[[-30, 30], [150, 210]], set_labels=True,
                              xlim=[-45, 360], ylim=[0, 45000], alpha=0.8,
			      colors=['k', 'b', 'g', 'orange'], kp_lim=[0.0, 9.0]):

    """ Plots histograms of 1-min imf(from omni) clock angles for the given clock angle bins
    """

    # make a db connection
    conn = sqlite3.connect(dbdir + dbname, detect_types = sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    if ax is None:
        fig, ax = plt.subplots()
    
    # Loop through each imf clock angle bin
    for i, bn in enumerate(bins):
	input_table = "b" + str((bn[0]%360)) + "_b" + str(bn[1]%360) + \
		      "_before" + str(before_mins) + "_after" +  str(after_mins) + \
		      "_bvec" + str(bvec_max).split('.')[-1] + "_all"

	command = "SELECT clock_angle, datetime FROM {tb} " + \
                  "WHERE (datetime BETWEEN '{stm}' AND '{etm}') AND " +\
                  "(kp BETWEEN {kp_low} AND {kp_high})"
	command = command.format(tb=input_table, stm=stm, etm=etm, kp_low=kp_lim[0], kp_high=kp_lim[1])
	cur.execute(command)
	rws = cur.fetchall()
	rws = [x[0] for x in rws]

        # Wrap the tailing points that exceed 0 or 360
        #if bn[0] < 0:
        if np.mean(bn) <= sector_center_dist:
            rws = [x if x < 3*sector_center_dist else x - 360 for x in rws] 
        if np.mean(bn) >= 360-sector_center_dist:
            rws = [x if x > np.mean(bn)-2*sector_center_dist else x + 360 for x in rws] 

	# plot the histogram
	bn_width = bn[1] - bn[0]
	nbins = (360 + bn_width)/5 - 1 
	ax.hist(rws, bins=nbins, range=(-bn_width, 360), color=colors[i], alpha=alpha)
	#ax.hist(rws, bins=71, range=(-60, 360), color=colors[i], alpha=0.6)
	
        # plot the boundary of each bins
	xs = [bn[0], (bn[0]+bn[1])/2., bn[1]]
	#ax.axvline(x=xs[0], color=colors[i], linestyle="--", linewidth=0.5)
	ax.axvline(x=xs[1], color='r', linestyle="-", linewidth=0.5)
	#ax.axvline(x=xs[2], color=colors[i], linestyle="--", linewidth=0.5)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # set lables
    if set_labels:
        ax.set_xlabel("IMF Clock Angle [degree]")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of IMF Clock Angle under Stable IMF Condition", fontsize="medium")
        ax.set_xticks(range(-sector_center_dist, 360+sector_center_dist, sector_center_dist))

    return

def plot_binned_imf_B_hist(stm, etm, dbdir, dbname, bvec_max=0.95, before_mins=20, 
                          after_mins=10, del_tm=10, sector_center_dist = 90,
                          imf_bin=[-30, 30], axes=None, params=["Bx", "By", "Bz"],
                          xlim=[-20, 20], ylim=[0, 45000], alpha=0.8, set_labels=True,
                          colors=['k', 'b', 'g'], kp_lim=[0.0, 9.0]):

    """ Plots histograms of 1-min imf Bs (Bx, By, Bz) from omni for a given clock angle bin
    """

    # make a db connection
    conn = sqlite3.connect(dbdir + dbname, detect_types = sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    if axes is None:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,6))
        axes = [ax for sublst in axes for ax in sublst]
    plt.gcf().subplots_adjust(hspace=0.5)
    
    # Loop through each imf clock angle bin
    input_table = "b" + str((imf_bin[0]%360)) + "_b" + str(imf_bin[1]%360) + \
                  "_before" + str(before_mins) + "_after" +  str(after_mins) + \
                  "_bvec" + str(bvec_max).split('.')[-1] + "_all"

    command = "SELECT Bx, By, Bz FROM {tb} " + \
              "WHERE (datetime BETWEEN '{stm}' AND '{etm}') AND " +\
              "(kp BETWEEN {kp_low} AND {kp_high})"
    command = command.format(tb=input_table, stm=stm, etm=etm, 
                             kp_low=kp_lim[0], kp_high=kp_lim[1])
    cur.execute(command)
    rws = cur.fetchall()

    imf_bin_width = imf_bin[1] - imf_bin[0]
    nbins = [40, 40, 40]
    rngs = [[-20, 20], [-20, 20], [-20, 20]]
    xlabels = {"Bx":"Bx [nT]", "By":"By [nT]", "Bz":"Bz [nT]"}
    titles = params

    rw_i_dct = {"Bx":0, "By":1, "Bz":2}  # The key-value pairs should match with the params order after SELECT
    for i in range(len(params)):
        ax = axes[i]
        var = [x[rw_i_dct[params[i]]] for x in rws]

	# plot the histogram
	ax.hist(var, bins=nbins[i], range=rngs[i], color=colors[i], alpha=alpha)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # set lables
        if set_labels:
            ax.set_xlabel(xlabels[params[i]])
            ax.set_ylabel("Count")
            ax.set_title("Histogram of IMF " + titles[i], fontsize="medium")


    return


def plot_binned_imf_hist(stm, etm, dbdir, dbname, bvec_max=0.95, before_mins=20, 
			      after_mins=10, del_tm=10, sector_center_dist = 90,
			      imf_bin=[-30, 30], axes=None,
			      colors=['k', 'b', 'g', 'orange'], kp_lim=[0.0, 9.0]):

    """ Plots histograms of 1-min imf Bs (Bx, By, Bz) and
    theta (from omni) for a given clock angle bin
    """

    # make a db connection
    conn = sqlite3.connect(dbdir + dbname, detect_types = sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    if axes is None:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
    plt.gcf().subplots_adjust(hspace=0.5)
    axes = [ax for sublst in axes for ax in sublst]
    
    # Loop through each imf clock angle bin
    input_table = "b" + str((imf_bin[0]%360)) + "_b" + str(imf_bin[1]%360) + \
                  "_before" + str(before_mins) + "_after" +  str(after_mins) + \
                  "_bvec" + str(bvec_max).split('.')[-1] + "_all"

    command = "SELECT * FROM {tb} " + \
              "WHERE (datetime BETWEEN '{stm}' AND '{etm}') AND " +\
              "(kp BETWEEN {kp_low} AND {kp_high})"
    command = command.format(tb=input_table, stm=stm, etm=etm, 
                             kp_low=kp_lim[0], kp_high=kp_lim[1])
    cur.execute(command)
    rws = cur.fetchall()

    imf_bin_width = imf_bin[1] - imf_bin[0]
    nbins = [40, 40, 40, (360 + imf_bin_width)/5 - 1]
    rngs = [[-20, 20], [-20, 20], [-20, 20], (-imf_bin_width, 360)]
    xlabels = ["Bx [nT]", "By [nT]", "Bz [nT]", "IMF Clock Angle [Degree]"]
    titles = ["Bx", "By", "Bz", "Clock Angle"]

    for i in range(1, 5):
        ax = axes[i-1]
        var = [x[i] for x in rws]
        if i == 4:
            if imf_bin[0] < 0:
                var = [x if x < sector_center_dist else x - 360 for x in var] 

	# plot the histogram
	ax.hist(var, bins=nbins[i-1], range=rngs[i-1], color=colors[i-1], alpha=0.8)
	
        # set lables
        ax.set_xlabel(xlabels[i-1])
        ax.set_ylabel("Count")
        ax.set_title("Histogram of IMF " + titles[i-1], fontsize="medium")
        if i == 4:
            ax.set_xlim([-sector_center_dist, 360])
            ax.set_ylim([0, 45000])
            ax.set_xticks(range(-sector_center_dist, 360+sector_center_dist, sector_center_dist))
        else:
            ax.set_xlim([-20, 20])
            ax.set_ylim([0, 60000])

    return

if __name__ == "__main__":

    import datetime as dt

    stm = dt.datetime(2011, 1, 1)
    etm = dt.datetime(2017, 1, 1)

    dbdir = "../../data/sqlite3/" 
    dbname = "binned_imf.sqlite"
    fig_path = "../plots/gmi_imf/"

    bvec_max = 0.95
    before_mins=50
    after_mins=0
    del_tm=10

#    bvec_max = 0.85
#    before_mins=80
#    after_mins=0
#    del_tm=10

    kp_lim = [0.0, 2.3]
    #kp_lim = [0.0, 9.0]
    #kp_lim = [2.6, 5.0]
    colors=['k', 'b', 'g', 'orange']


    sector_center_dist = 90
    # set bins for all clock angle range
#    sector_width = 60
#    bins = [[x-sector_width/2, x+sector_width/2] for x in np.arange(0, 360, sector_center_dist)]
    # set bins for IMF clock angle near 90 or 270
    sector_width = 30 
    sector_centers = [80 - sector_width/2, 100 + sector_width/2, 
                      260 - sector_width/2, 280 + sector_width/2]
    bins = []
    for ctr in sector_centers:
        bins.append([ctr - sector_width/2, ctr + sector_width/2])

    #bins = [[50, 80]]

    #################################################
#    # Plot the histogram of IMF for stable period
#    fig_name = "hist_imf_clock_angle_" + \
#               stm.strftime("%Y%m%d") + "_" + etm.strftime("%Y%m%d") +\
#               "_before" + str(before_mins) + "_after" +  str(after_mins) + \
#               "_bvec" + str(bvec_max).split('.')[-1] + "_kp_" +\
#               "_to_".join(["".join(str(x).split('.')) for x in kp_lim]) + ".png"
#    # plot the histogram of binned imf_clock_angle
#    fig = plot_binned_imf_clock_angle_hist(stm, etm, dbdir, dbname, bvec_max=bvec_max,
#				    before_mins=before_mins, after_mins=after_mins,
#				    del_tm=del_tm, sector_center_dist=sector_center_dist,
#                                    bins=bins, colors=colors, kp_lim=kp_lim)
#    fig.savefig(fig_path+fig_name, dpi=200)


    #################################################
    # Plot the hist of imf params in a given imf bin
    for bn in bins:
        fig_name = "binned_imf_hist_" + \
                   stm.strftime("%Y%m%d") + "_" + etm.strftime("%Y%m%d") +\
                   "_b" + str((bn[0]%360)) + "_b" + str(bn[1]%360) + \
                   "_before" + str(before_mins) + "_after" +  str(after_mins) + \
                   "_bvec" + str(bvec_max).split('.')[-1] + "_kp_" +\
                   "_to_".join(["".join(str(x).split('.')) for x in kp_lim]) + ".png"
        fig = plot_binned_imf_hist(stm, etm, dbdir, dbname, bvec_max=bvec_max,
                                 before_mins=before_mins, after_mins=after_mins,
                                 del_tm=del_tm, sector_center_dist=sector_center_dist,
                                 imf_bin=bn, colors=colors,
                                 kp_lim=kp_lim)

        fig.savefig(fig_path+fig_name, dpi=200)

#    #################################################
#    # Plot the histogram of all the imf for the interval between stm and etm
#    dbname = "gmi_imf.sqlite"
#    #param = "theta"
#    param = "Bx"
#    fig_name = "hist_all_imf_" + param + "_" + \
#               stm.strftime("%Y%m%d") + "_" + etm.strftime("%Y%m%d") + ".png"
#    fig = plot_imf_hist(stm, etm, dbdir, dbname, param=param)
#    fig.savefig(fig_path+fig_name, dpi=200)

