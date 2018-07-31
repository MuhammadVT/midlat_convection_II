import matplotlib
matplotlib.use('Agg')
import sqlite3
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import datetime as dt
import numpy as np
from matplotlib.ticker import MultipleLocator
from imf_hist import plot_binned_imf_clock_angle_hist, plot_binned_imf_B_hist
import sys
sys.path.append("../")
from velcomp_vs_time import plot_center_axis

stm = dt.datetime(2011, 1, 1)
etm = dt.datetime(2017, 1, 1)

seasons = "winter"
kp_lim = [0.0, 2.3]

dbdir = "../../data/sqlite3/" 
dbname = "binned_imf.sqlite"
fig_path = "../plots/gmi_imf/hist_of_imf_binned_by_clock_angle/"

bvec_max = 0.90
before_mins=50
after_mins=0
del_tm=10

# Create clock angle bins
sector_center_dist = 45
sector_width = 40
# set bins for all clock angle ranges
imf_bins = [[x-sector_width/2, x+sector_width/2] for x in np.arange(0, 360, sector_center_dist)]
#imf_bins = [[-30, 30] for x in imf_bins]

# Determines how to place the imf_bins into panels,
# NOTE: must match with imf_bins
ax_idxs = [1, 2, 5, 8, 7, 6, 3, 0]
bins_txt = ["Bz+", "By+/Bz+", "By+", "By+/Bz-",
	    "Bz-", "By-/Bz-", "By-", "By-/Bz+"]

## set bins for IMF clock angle near 90 or 270
#sector_centers = [80 - sector_width/2, 100 + sector_width/2,
#                  260 - sector_width/2, 280 + sector_width/2]
#imf_bins = []
#for ctr in sector_centers:
#    imf_bins.append([ctr - sector_width/2, ctr + sector_width/2])
##bins_txt = ["Bz+", "By+", "Bz-", "By-"]
#bins_txt = ["By+, Bz+", "By+, Bz-", "By-, Bz-", "By-, Bz+"]

#colors=['k', 'b', 'g', 'orange']

def clock_angle_hist(all_in_one_axis=False):
    """
    Plots the hist of clock angles,
    NOTE: one IMF bin per panel if all_in_one_axis is False
    """

    colors=['k'] * len(imf_bins)
    xlim=[-45, 360]
    ylim=[0, 35000]
    # create subplots
    if all_in_one_axis:
	fig, ax = plt.subplots()
	# plot the histogram of binned imf_clock_angle
	plot_binned_imf_clock_angle_hist(stm, etm, dbdir, dbname, ax=ax, bvec_max=bvec_max,
					before_mins=before_mins, after_mins=after_mins,
					del_tm=del_tm, sector_center_dist=sector_center_dist,
					bins=imf_bins, colors=colors, kp_lim=kp_lim, alpha=1.0)

        fig_name = "hist_imf_clock_angle_v1_" + \
                   stm.strftime("%Y%m%d") + "_" + etm.strftime("%Y%m%d") +\
                   "_before" + str(before_mins) + "_after" +  str(after_mins) + \
                   "_bvec" + str(bvec_max).split('.')[-1] + "_kp_" +\
                   "_to_".join(["".join(str(x).split('.')) for x in kp_lim])

    else:
	fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,6),
				 sharex=True, sharey=True)
	axes = [x for subls in axes for x in subls]
	fig.subplots_adjust(hspace=0.3)

	for i, imf_bin in enumerate(imf_bins):
	    ax = axes[ax_idxs[i]]
	    # plot the histogram of binned imf_clock_angle
	    plot_binned_imf_clock_angle_hist(stm, etm, dbdir, dbname, ax=ax, bvec_max=bvec_max,
					    before_mins=before_mins, after_mins=after_mins,
					    del_tm=del_tm, sector_center_dist=sector_center_dist,
                                            xlim=xlim, ylim=ylim,
					    bins=[imf_bin], colors=colors, kp_lim=kp_lim, set_labels=False)
            ax.set_title("Hist. of IMF Clock Angle", fontsize=10)

        # Plot the center axis for IMF clock angle
        plot_center_axis(axes[4], sector_center_dist=sector_center_dist,
                         sector_width=sector_width, xlim=xlim, ylim=ylim)

         # Set axis labels
        # Add label to first column and last row
        for i in [0, 3, 6]:
            axes[i].set_ylabel("Count", fontsize=9)
            axes[i].yaxis.set_major_locator(MultipleLocator(base=10000))
        for i in range(6,9):
            axes[i].set_xlabel("IMF Clock Angle [Degree]", fontsize=9)
            axes[i].xaxis.set_major_locator(MultipleLocator(base=sector_center_dist))
            # Set tick fontsize
            for tick in axes[i].xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
                tick.label.set_rotation(30)

        # Save the figure
        fig_name = "hist_imf_clock_angle_v2_" + \
                   stm.strftime("%Y%m%d") + "_" + etm.strftime("%Y%m%d") +\
                   "_before" + str(before_mins) + "_after" +  str(after_mins) + \
                   "_bvec" + str(bvec_max).split('.')[-1] + "_kp_" +\
                   "_to_".join(["".join(str(x).split('.')) for x in kp_lim])

    fig.savefig(fig_path+fig_name + ".png", dpi=200, bbox_inches="tight")
    #fig.savefig(fig_path+fig_name + ".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return

def IMF_B_hist():

    """
    Plots the hist of certain IMF component,
    """
    param_list = ["By", "Bz"]
    colors=['k'] * len(imf_bins)
    xlim=[-20, 20]
    ylim=[0, 120000]

    for param in param_list:
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,6),
                                 sharex=True, sharey=True)
        axes = [x for subls in axes for x in subls]
        fig.subplots_adjust(hspace=0.3)

        for i, imf_bin in enumerate(imf_bins):
            ax = axes[ax_idxs[i]]
            # Plot the hist of imf B in a given imf bin
            plot_binned_imf_B_hist(stm, etm, dbdir, dbname, axes=[ax], bvec_max=bvec_max,
                                   before_mins=before_mins, after_mins=after_mins,
                                   del_tm=del_tm, sector_center_dist=sector_center_dist,
                                   imf_bin=imf_bin, params=[param], colors=colors, xlim=xlim, ylim=ylim,
                                   kp_lim=kp_lim, set_labels=False)
            ax.set_title("Hist. of IMF " + param, fontsize=10)

        # Plot the center axis for IMF clock angle
        plot_center_axis(axes[4], sector_center_dist=sector_center_dist,
                         sector_width=sector_width, xlim=xlim, ylim=ylim)

         # Set axis labels
        # Add label to first column and last row
        for i in [0, 3, 6]:
            axes[i].set_ylabel("Count", fontsize=9)
            axes[i].yaxis.set_major_locator(MultipleLocator(base=20000))
        for i in range(6,9):
            axes[i].set_xlabel(param + " [nT]", fontsize=9)
            axes[i].xaxis.set_major_locator(MultipleLocator(base=4))
            # Set tick fontsize
            for tick in axes[i].xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
                #tick.label.set_rotation(30)


        # Save figure
        fig_name = "binned_imf_" + param + "_hist_" + \
                   stm.strftime("%Y%m%d") + "_" + etm.strftime("%Y%m%d") +\
                   "_before" + str(before_mins) + "_after" +  str(after_mins) + \
                   "_bvec" + str(bvec_max).split('.')[-1] + "_kp_" +\
                   "_to_".join(["".join(str(x).split('.')) for x in kp_lim])

        fig.savefig(fig_path+fig_name + ".png", dpi=200, bbox_inches="tight")
        #fig.savefig(fig_path+fig_name + ".pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)

    return

def IMF_B_theta_hist():

    """
    Plots the hist of IMF components and the clock angle,
    """
    param_list = ["By", "Bz"]
    colors=['k'] * len(imf_bins)
    B_xlim=[-16, 16]
    B_ylim=[0, 120000]
    theta_xlim=[-45, 360]
    theta_ylim=[0, 35000]

    # Create a figure
    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(12,5), sharey=True)
    fig.subplots_adjust(hspace=0.3)

    # Plot hist of IMF components defined in param_list
    for i, param in param_list:
        for j, imf_bin in enumerate(imf_bins):
            ax = axes[i, ax_idxs[j]]
            if i <=1:
                # Plot the hist of imf B in a given imf bin
                plot_binned_imf_B_hist(stm, etm, dbdir, dbname, axes=[ax], bvec_max=bvec_max,
                                       before_mins=before_mins, after_mins=after_mins,
                                       del_tm=del_tm, sector_center_dist=sector_center_dist,
                                       imf_bin=imf_bin, params=[param], colors=colors,
                                       xlim=B_xlim, ylim=B_ylim,
                                       kp_lim=kp_lim, set_labels=False)
                #ax.set_title("Hist. of IMF " + param, fontsize=10)

                # Set axis labels
                ax.set_ylabel("Count", fontsize=9)
                ax.yaxis.set_major_locator(MultipleLocator(base=20000))
                ax.set_xlabel(param + " [nT]", fontsize=9)
                ax.xaxis.set_major_locator(MultipleLocator(base=4))
            else:
                # plot the histogram of binned imf_clock_angle
                plot_binned_imf_clock_angle_hist(stm, etm, dbdir, dbname, ax=ax, bvec_max=bvec_max,
                                                before_mins=before_mins, after_mins=after_mins,
                                                del_tm=del_tm, sector_center_dist=sector_center_dist,
                                                xlim=theta_xlim, ylim=theta_ylim,
                                                bins=[imf_bin], colors=colors,
                                                kp_lim=kp_lim, set_labels=False)
                # Set axis labels
                ax.set_title("Hist. of IMF Clock Angle", fontsize=10)
                ax.yaxis.set_major_locator(MultipleLocator(base=10000))
                ax.set_xlabel("Clock Ang. [Deg]", fontsize=9)
                ax.xaxis.set_major_locator(MultipleLocator(base=sector_center_dist))

            # Set tick fontsize
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
                #tick.label.set_rotation(30)

    # Set titles for each column
    for j in range(8):
        ax = axes[0, ax_idxs[j]].set_title(bins_txt[j], fontsize=10)

    # Save figure
    fig_name = "binned_imf_hist_" + \
               stm.strftime("%Y%m%d") + "_" + etm.strftime("%Y%m%d") +\
               "_before" + str(before_mins) + "_after" +  str(after_mins) + \
               "_bvec" + str(bvec_max).split('.')[-1] + "_kp_" +\
               "_to_".join(["".join(str(x).split('.')) for x in kp_lim])

    fig.savefig(fig_path+fig_name + ".png", dpi=200, bbox_inches="tight")
    #fig.savefig(fig_path+fig_name + ".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return


if __name__ == "__main__":
    #clock_angle_hist(all_in_one_axis=False)
    IMF_B_hist()
