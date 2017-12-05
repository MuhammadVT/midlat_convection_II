import matplotlib
matplotlib.use('Agg')

import sqlite3
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def plot_imf_clock_angle_hist(stm, etm, dbdir, dbname, bvec_max=0.95, before_mins=20, 
			      after_mins=10, del_tm=10, sector_center_dist = 90,
			      bins=[[-30, 30], [150, 210]],
			      colors=['k', 'b', 'g', 'orange']):

    """ Plots histograms of imf clock angles for the given clock angle bins
    """

    # make a db connection
    conn = sqlite3.connect(dbdir + dbname, detect_types = sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    fig, ax = plt.subplots()
    
    # Loop through each imf clock angle bin
    for i, bn in enumerate(bins):
	input_table = "b" + str((bn[0]%360)) + "_b" + str(bn[1]%360) + \
		      "_before" + str(before_mins) + "_after" +  str(after_mins) + \
		      "_bvec" + str(bvec_max).split('.')[-1] + "_all"

	command = "SELECT clock_angle, datetime FROM {tb} " + \
                  "WHERE datetime BETWEEN '{stm}' AND '{etm}' "
	command = command.format(tb=input_table, stm=stm, etm=etm)
	cur.execute(command)
	rws = cur.fetchall()
	rws = [x[0] for x in rws]
        if bn[0] < 0:
            rws = [x if x < sector_center_dist else x - 360 for x in rws] 

	# plot the histogram
	bn_width = bn[1] - bn[0]
	nbins = (360 + bn_width)/5 - 1 
	ax.hist(rws, bins=nbins, range=(-bn_width, 360), color=colors[i], alpha=0.8)
	#ax.hist(rws, bins=71, range=(-60, 360), color=colors[i], alpha=0.6)
	
        # plot the boundary of each bins
	xs = [bn[0], (bn[0]+bn[1])/2., bn[1]]
	#ax.axvline(x=xs[0], color=colors[i], linestyle="--", linewidth=0.5)
	ax.axvline(x=xs[1], color='r', linestyle="-", linewidth=0.5)
	#ax.axvline(x=xs[2], color=colors[i], linestyle="--", linewidth=0.5)

    # set lables
    ax.set_xlabel("IMF Clock Angle [degree]")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of IMF Clock Angle under Stable IMF Condition", fontsize="medium")
    ax.set_xlim([-sector_center_dist, 360])
    ax.set_xticks(range(-sector_center_dist, 360+sector_center_dist, sector_center_dist))

    return fig

if __name__ == "__main__":

    import datetime as dt
    import numpy as np

    stm = dt.datetime(2011, 1, 1)
    etm = dt.datetime(2017, 1, 1)

    dbdir = "../../data/sqlite3/"
    dbname = "binned_imf.sqlite"
    bvec_max = 0.95
    before_mins=20
    after_mins=10
    del_tm=10
    colors=['k', 'b', 'g', 'orange']
    
    # set the bins
    sector_width = 60
    sector_center_dist = 90
    bins = [[x-sector_width/2, x+sector_width/2] for x in np.arange(0, 360, sector_center_dist)]
    fig = plot_imf_clock_angle_hist(stm, etm, dbdir, dbname, bvec_max=bvec_max,
				    before_mins=before_mins, after_mins=after_mins,
				    del_tm=del_tm, sector_center_dist=sector_center_dist,
                                    bins=bins, colors=colors)

    fig_path = "../plots/gmi_imf/"
    fig_name = "hist_imf_clock_angle_" + stm.strftime("%Y%m%d") + "_" + etm.strftime("%Y%m%d") + ".png"
    fig.savefig(fig_path+fig_name, dpi=200)

