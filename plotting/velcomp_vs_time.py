import matplotlib
matplotlib.use('Agg')

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
from convection import fetch_data
from matplotlib.ticker import MultipleLocator
import numpy as np


def vel_vs_lt(ax, data_dict, veldir="zonal", center_at_zero_mlt=True,
               glatc_list=None, title="xxx", add_err_bar=False,
               color_list=None, marker_size=2, marker_type="o",
               xlim=[-6, 6], ylim=[-80, 30]):
    
    """ plots the flow vectors in local time (MLT or SLT) coords

    parameters
    ----------
    veldir : str
        veocity component. if set to "all" then it means the velocity magnitude
    """

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.collections import PolyCollection,LineCollection
    import numpy as np

    # calculate velocity components
    vel_mag = data_dict['vel_mag']
    vel_dir = np.deg2rad(data_dict['vel_dir'])
    vel_mag_err = data_dict['vel_mag_err']

    if veldir == "zonal":
        vel_comp = vel_mag*(-1.0)*np.sin(vel_dir)
        vel_comp_err = vel_mag_err*(-1.0)*np.sin(vel_dir)
    elif veldir == "meridional":
        vel_comp = vel_mag*(-1.0)*np.cos(vel_dir)
        vel_comp_err = vel_mag_err*(-1.0)*np.cos(vel_dir)
    elif veldir == "all":
        vel_comp = np.abs(vel_mag)
        vel_comp_err = vel_mag_err
    vel_mlt = data_dict['glonc'] / 15.
    
    # colors of different lines
    if color_list is None:
        color_list = ['darkblue', 'b', 'dodgerblue', 'c', 'g', 'orange', 'r']
    color_list.reverse()

    # MLATs
    if glatc_list is None:
        glatc_list = np.array([50.5])
    for jj, mlat in enumerate(glatc_list):
        vel_comp_jj = [vel_comp[i] for i in range(len(vel_comp)) if data_dict['glatc'][i] == mlat]
        vel_mlt_jj = [vel_mlt[i] for i in range(len(vel_comp)) if data_dict['glatc'][i] == mlat]
        vel_comp_err_jj = [vel_comp_err[i] for i in range(len(vel_comp_err)) if data_dict['glatc'][i] == mlat]
        if center_at_zero_mlt:
            # center at 0 MLT
            vel_mlt_jj = [x if x <=12 else x-24 for x in vel_mlt_jj]
        # plot the velocities for each MLAT
        ax.scatter(vel_mlt_jj, vel_comp_jj, c=color_list[jj],
                #marker='o', s=3, linewidths=.5, edgecolors='face', label=str(int(mlat)))
                marker=marker_type, s=marker_size, linewidths=.5, edgecolors='face', label=str(mlat))

        if add_err_bar:
            ax.errorbar(vel_mlt_jj, vel_comp_jj, yerr=vel_comp_err_jj, mfc=color_list[jj],
                    #marker='o', s=3, linewidths=.5, edgecolors='face', label=str(int(mlat)))
                    fmt=marker_type, ms=marker_size, elinewidth=.5, mec=color_list[jj], ecolor="k")

    # add text
    ax.set_title(title, fontsize="small")

    # Set xtick directions
    ax.tick_params(direction="in")

    # Set ytick format
    ax.yaxis.set_major_locator(MultipleLocator(20))

    # add zero-line
    if veldir != "all":
        ax.axhline(y=0, color='k', linewidth=0.7)

    # set axis limits
    if center_at_zero_mlt:
        ax.set_xlim(xlim)
        # add legend
        #ax.legend(bbox_to_anchor=(1.01, 0.91), fontsize=6)
        #ax.legend(loc="center right", fontsize=6)
    else:
        ax.set_xlim(xlim)
        # add legend
        #ax.legend(loc='center right', fontsize=8)
        #ax.legend(bbox_to_anchor=(0.65, 0.96), fontsize=8)

    if veldir == "all":
        ax.set_ylim(ylim)
    else:
        #ax.set_ylim([-30, 30])
        ax.set_ylim(ylim)
    
    return

def by_season():

    # input parameters
    #nvel_min=100
    nvel_min=300
    #del_lat=3
    del_lat=1
    #lat_range=[58, 65]
    lat_range=[52, 59]

    glatc_list = np.arange(lat_range[1]-0.5, lat_range[0]-0.5, -del_lat)
    #lat_range=[40, 60]
    #glatc_list = np.array([41.5, 50.5])
    if len(glatc_list) == 0:
        glatc_list = np.array([lat_range[0]]+0.5)
    
    #add_err_bar = True
    add_err_bar = False
    ftype = "fitacf"
    coords = "mlt"

    #veldir="all"
    #veldir="zonal"
    veldir="meridional"
    center_at_zero_mlt=True
    #center_at_zero_mlt=False
    xlim=[-6, 6]
    ylim=[-80, 30]

    seasons = ["winter", "summer", "equinox"]
    #seasons = ["winter"]

    sqrt_weighting = True
    rads_txt = "six_rads"
    #rads_txt = "ade_adw"
    #rads_txt = "hok_hkw"

    #years = [2011, 2012]
    #years_txt = "_years_" + "_".join([str(x) for x in years])
    years_txt = ""

    #kp_text = "_kp_00_to_23"
    kp_text = "_kp_00_to_03"
    kp_text_dict ={"_kp_00_to_03" : r", Kp = 0",
                   "_kp_07_to_13" : r", Kp = 1",
                   "_kp_17_to_23" : r", Kp = 2",
                   "_kp_27_to_33" : r", Kp = 3",
                   "_kp_27_to_43" : r", 3-$\leq$Kp$\leq$4+",
                   "_kp_37_to_90" : r", Kp $\geq$ 4-"}

    #input_table = "master_cosfit_hok_hkw_kp_00_to_23_azbin_nvel_min_5"
    input_table = "master_cosfit_" + rads_txt + kp_text + years_txt

    fig_dir = "./plots/velcomp_vs_time/" + kp_text[1:] + "/data_in_mlt/"
    if center_at_zero_mlt:
        fig_name = rads_txt + years_txt + "_seasonal_" + veldir+ "_vel_vs_ltm_c0" +\
                   "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1])
    else:
        fig_name = rads_txt + years_txt + "_seasonal_" + veldir+ "_vel_vs_ltm" +\
                   "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1])

    # create subplots
    fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=None, sharex=True)
    fig.subplots_adjust(hspace=0.3)

    if len(seasons) == 1:
        axes = [axes]

    for i, season in enumerate(seasons):
        # fetches the data from db 
        data_dict = fetch_data(input_table, lat_range=lat_range,
                    nvel_min=nvel_min, season=season,
                    config_filename="../mysql_dbconfig_files/config.ini",
                    section="midlat", db_name=None, ftype=ftype,
                    coords=coords, sqrt_weighting=sqrt_weighting)

        # plot the flow vector components
        if veldir == "all" :
            title = "Velocity Magnitude, " + season[0].upper()+season[1:] + kp_text_dict[kp_text]
        else:
            title = veldir[0].upper()+veldir[1:] + " Velocities, " +\
                    season[0].upper()+season[1:] + kp_text_dict[kp_text]
        vel_vs_lt(axes[i], data_dict, veldir=veldir, center_at_zero_mlt=center_at_zero_mlt,
                  glatc_list=glatc_list, title=title, add_err_bar=add_err_bar,
                  xlim=xlim, ylim=ylim)

    # set axis label
    axes[-1].set_xlabel("MLT")
    #axes[-1].set_xlabel("Solar Local Time")
    axes[-1].xaxis.set_major_locator(MultipleLocator(base=3))
    if center_at_zero_mlt:
        xlabels = [item.get_text() for item in axes[-1].get_xticklabels()]
        #xlabels = [str(x) for x in range(12, 24, 3) + range(0, 15, 3)]
        #plt.xticks(range(-12, 15, 3), xlabels)
        xlabels = [str(x) for x in range(18, 24, 3) + range(0, 9, 3)]
        plt.xticks(range(-6, 9, 3), xlabels)

    # save the fig
    fig.savefig(fig_dir + fig_name + ".png", dpi=300)
    #plt.show()

    return

def six_rads_by_year():

    # input parameters
    nvel_min=300
    del_lat=1
    lat_range=[52, 61]
    #lat_range=[52, 61]
    glatc_list = np.arange(lat_range[0]+0.5, lat_range[1]+0.5, del_lat)
    if len(glatc_list) == 0:
        glatc_list = np.array([lat_range[0]]+0.5)
    
    #add_err_bar = True
    add_err_bar = False
    ftype = "fitacf"
    coords = "mlt"

    #veldir="all"
    #veldir="zonal"
    veldir="meridional"
    center_at_zero_mlt=True
    #center_at_zero_mlt=False
    sqrt_weighting = True

    seasons = ["winter", "equinox", "summer"]
    rads_txt = "six_rads"
    years_list = [[2011, 2012], [2013, 2014], [2015, 2016], None]
    legend_txt = ["11_12", "13_14", "15_16", "11_16"]

#    rads_txt = "ade_adw"
#    years_list = [[2013, 2014], [2015, 2016], None]
#    legend_txt = ["13_14", "15_16", "13_16"]

    for season in seasons:

	fig_dir = "./plots/velcomp_vs_time/kp_l_3/data_in_mlt/"
	if center_at_zero_mlt:
	    fig_name = rads_txt + "_by_year_" + season + "_" + veldir + "_vel_vs_ltm_c0" +\
		       "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1])
	else:
	    fig_name = rads_txt + "_by_year_" + "_" + season + "_" + veldir + "_vel_vs_ltm_c0" +\
		       "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1])

	# create subplots
	fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,6), 
				 sharex=True, sharey=True)
	axes = [ax for l in axes for ax in l]
	fig.subplots_adjust(hspace=0.4)

	# fetches the data from db 
	data_dict_list = []
	for i in range(len(years_list)):
	    years = years_list[i]
	    if years is None:
		years_txt = ""
	    else:
		years_txt = "_years_" + "_".join([str(x) for x in years])
	    input_table = "master_cosfit_" + rads_txt + "_kp_00_to_23" + years_txt
	    data_dict = fetch_data(input_table, lat_range=lat_range,
			nvel_min=nvel_min, season=season,
			config_filename="../mysql_dbconfig_files/config.ini",
			section="midlat", db_name=None, ftype=ftype,
			coords=coords, sqrt_weighting=sqrt_weighting)
	    data_dict_list.append(data_dict)

	#color_list = ['darkblue', 'b', 'dodgerblue', 'c', 'g', 'orange', 'r']
	color_list = ['k', 'dodgerblue', 'g', 'r']
	#color_list = ['k', 'g', 'r']
	#markers = ['o', '+', '*', '.']
	for i, latc in enumerate(glatc_list):
	    ax = axes[i]
	    # plot the flow vector components for each latitude
	    if veldir == "all" :
		title = "Velocity Magnitude, " + season[0].upper()+season[1:] + r", Kp $\leq$ 2+"
	    else:
		title = veldir[0].upper()+veldir[1:] + " Vel, " +\
			season[0].upper()+season[1:] + ", MLAT=" + str(latc) +\
			r", Kp $\leq$ 2+"
	    for j in range(len(years_list)):
		vel_vs_lt(ax, data_dict_list[j], veldir=veldir,
			  center_at_zero_mlt=center_at_zero_mlt,
			  glatc_list=[latc], title=title, add_err_bar=add_err_bar,
			  color_list=[color_list[j]], marker_size=0.8)

	    # change the font
	    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
			 ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_fontsize(8)
	    ax.legend().set_visible(False)

	# remove labels
	for ax in axes:
	    ax.set_xlabel("")
	    ax.set_ylabel("")

	# add label to last row
	for i in range(6,9):
	    axes[i].set_xlabel("MLT")
	    #axes[i].set_xlabel("Solar Local Time")
	    axes[i].xaxis.set_major_locator(MultipleLocator(base=3))

	# add legend
	#axes[2].legend(bbox_to_anchor=(1.05, 1.00), fontsize=6)
	lg = axes[2].legend()
	txts = lg.get_texts()
	for i in range(len(years_list)):
	    txts[i].set_text(legend_txt[i])
	    txts[i].set_fontsize(9)
	lg.set_bbox_to_anchor((1.02, 0.93))

	# set axis label
	if center_at_zero_mlt:
	    xlabels = [item.get_text() for item in axes[-1].get_xticklabels()]
	    xlabels = [str(x) for x in range(18, 24, 3) + range(0, 9, 3)]
	    plt.xticks(range(-6, 9, 3), xlabels)

	# save the fig
	fig.savefig(fig_dir + fig_name + ".png", dpi=300)
	#plt.show()

    return

def by_pairs_of_radars():

    # input parameters
    nvel_min=300
    del_lat=1
    #lat_range=[52, 58]
    lat_range=[52, 61]
    glatc_list = np.arange(lat_range[0]+0.5, lat_range[1]+0.5, del_lat)
    if len(glatc_list) == 0:
        glatc_list = np.array([lat_range[0]]+0.5)
    
    #add_err_bar = True
    add_err_bar = False
    ftype = "fitacf"
    coords = "mlt"

    #veldir="all"
    veldir="zonal"
    #veldir="meridional"
    center_at_zero_mlt=True
    #center_at_zero_mlt=False
    sqrt_weighting = True

    seasons = ["winter", "equinox", "summer"]
    #rads_txt_list = ["bks_wal", "fhe_fhw", "cve_cvw", "ade_adw", "six_rads"]
    #rads_txt_list = ["bks_wal", "fhe_fhw", "cve_cvw", "six_rads"]
    rads_txt_list = ["bks_wal", "fhe_fhw", "cve_cvw"]
    years_txt = ""

    for season in seasons:

	fig_dir = "./plots/velcomp_vs_time/kp_l_3/data_in_mlt/"
	if center_at_zero_mlt:
	    fig_name = "rad_pairs_v3_" + season + "_" + veldir + "_vel_vs_ltm_c0" +\
		       "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1])
	else:
	    fig_name = "rad_pairs_v3_" + season + "_" + veldir + "_vel_vs_ltm_c0" +\
		       "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1])

	# create subplots
	fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,6), 
				 sharex=True, sharey=True)
	axes = [ax for l in axes for ax in l]
	fig.subplots_adjust(hspace=0.4)

	# fetches the data from db 
	data_dict_list = []
	for i in range(len(rads_txt_list)):
	    input_table = "master_cosfit_" + rads_txt_list[i] + "_kp_00_to_23" + years_txt
	    data_dict = fetch_data(input_table, lat_range=lat_range,
			nvel_min=nvel_min, season=season,
			config_filename="../mysql_dbconfig_files/config.ini",
			section="midlat", db_name=None, ftype=ftype,
			coords=coords, sqrt_weighting=sqrt_weighting)
	    data_dict_list.append(data_dict)

	#color_list = ['darkblue', 'b', 'dodgerblue', 'c', 'g', 'orange', 'r']
	#color_list = ['k', 'dodgerblue', 'g', 'orange', 'r']
	#color_list = ['k', 'dodgerblue', 'g', 'r']
	color_list = ['k', 'g', 'r']
	#markers = ['o', '+', '*', '.']
	for i, latc in enumerate(glatc_list):
	    ax = axes[i]
	    # plot the flow vector components for each latitude
	    if veldir == "all" :
		title = "Velocity Magnitude, " + season[0].upper()+season[1:] + r", Kp $\leq$ 2+"
	    else:
		title = veldir[0].upper()+veldir[1:] + " Vel, " +\
			season[0].upper()+season[1:] + ", MLAT=" + str(latc) +\
			r", Kp $\leq$ 2+"
	    for j in range(len(rads_txt_list)):
		vel_vs_lt(ax, data_dict_list[j], veldir=veldir,
			  center_at_zero_mlt=center_at_zero_mlt,
			  glatc_list=[latc], title=title, add_err_bar=add_err_bar,
			  color_list=[color_list[j]], marker_size=0.8)

	    # change the font
	    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
			 ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_fontsize(8)
	    ax.legend().set_visible(False)

	# remove labels
	for ax in axes:
	    ax.set_xlabel("")
	    ax.set_ylabel("")

	# add label to last row
	for i in range(6,9):
	    axes[i].set_xlabel("MLT")
	    #axes[i].set_xlabel("Solar Local Time")
	    axes[i].xaxis.set_major_locator(MultipleLocator(base=3))

	# add legend
	#axes[2].legend(bbox_to_anchor=(1.05, 1.00), fontsize=6)
	lg = axes[2].legend()
	legend_txt = rads_txt_list
	txts = lg.get_texts()
	for i in range(len(rads_txt_list)):
	    txts[i].set_text(legend_txt[i])
	    txts[i].set_fontsize(9)
	lg.set_bbox_to_anchor((1.02, 0.93))

	# set axis label
	if center_at_zero_mlt:
	    xlabels = [item.get_text() for item in axes[-1].get_xticklabels()]
	    xlabels = [str(x) for x in range(18, 24, 3) + range(0, 9, 3)]
	    plt.xticks(range(-6, 9, 3), xlabels)

	# save the fig
	fig.savefig(fig_dir + fig_name + ".png", dpi=300)
	#plt.show()

    return

def by_kp(single_kp=True, single_lat=False):


    # input parameters
    nvel_min=100
    del_lat=1
    lat_range=[52, 59]
    glatc_list = np.arange(lat_range[1]-0.5, lat_range[0]-0.5, -del_lat)
    if len(glatc_list) == 0:
        glatc_list = np.array([lat_range[0]]+0.5)
    
    ftype = "fitacf"
    coords = "mlt"
    db_name = "master_" + coords + "_" + ftype
    #seasons = ["winter"]
    seasons = ["winter", "equinox", "summer"]
    #veldir="zonal"
    veldir="meridional"
    xlim=[-6, 6]
    ylim=[-80, 30]

    center_at_zero_mlt=True
    sqrt_weighting = True
    add_err_bar = False

    years_txt = ""
    rads_txt = "six_rads"
    # Used when single_kp is set to True
    kp_texts = ["_kp_00_to_03", "_kp_07_to_13", "_kp_17_to_23", "_kp_27_to_33"]
    # used when single_lat is set to True
    kp_text_list = ["_kp_00_to_03",      
                    "_kp_07_to_13",
                    "_kp_17_to_23",
                    "_kp_27_to_33"]
                    #"_kp_00_to_23"]

    kp_text_dict ={"_kp_00_to_03" : r", Kp = 0",
                   "_kp_00_to_23" : r", Kp$\leq$2+",
                   "_kp_07_to_13" : r", Kp = 1",
                   "_kp_17_to_23" : r", Kp = 2",
                   "_kp_27_to_33" : r", Kp = 3",
                   "_kp_27_to_43" : r", 3-$\leq$Kp$\leq$4+",
                   "_kp_37_to_90" : r", Kp $\geq$ 4-"}

    fig_dir = "./plots/velcomp_vs_time/kp_l_3/data_in_mlt/binned_by_kp/"
    if single_kp:
        # create subplots
        fig, axes = plt.subplots(nrows=3, ncols=len(kp_texts), figsize=(12,5),
                                 sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.3)
    
        for j, kp_text in enumerate(kp_texts):
            #input_table = "master_cosfit_hok_hkw_kp_00_to_23"
            #input_table = "master_cosfit_hok_hkw_kp_00_to_23_azbin_nvel_min_5"
            input_table = "master_cosfit_" + rads_txt + kp_text + years_txt

            data_dict_list = []
            for i, season in enumerate(seasons):
                ax = axes[i, j]
                # fetches the data from db 
                data_dict = fetch_data(input_table, lat_range=lat_range,
                            nvel_min=nvel_min, season=season,
                            config_filename="../mysql_dbconfig_files/config.ini",
                            section="midlat", db_name=db_name, ftype=ftype,
                            coords=coords, sqrt_weighting=sqrt_weighting)
                data_dict_list.append(data_dict)

                # Plot the flow vector components for each imf bin
                if veldir == "all" :
                    title = "Velocity Magnitude, " + season[0].upper()+season[1:] + kp_text_dict[kp_text]
                else:
                    title = veldir[0].upper()+veldir[1:] + " Flow, " +\
                            season[0].upper()+season[1:] + kp_text_dict[kp_text]
                vel_vs_lt(ax, data_dict_list[i], veldir=veldir,
                          center_at_zero_mlt=center_at_zero_mlt,
                          glatc_list=glatc_list, title=title, add_err_bar=add_err_bar,
                          color_list=None, marker_size=1)

                # change the font
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(9)
                ax.legend().set_visible(False)

        # set axis ticklabel
        if center_at_zero_mlt:
            xlabels = [item.get_text() for item in axes[2,2].get_xticklabels()]
            xlabels = [str(x) for x in range(18, 24, 3) + range(0, 9, 3)]
            plt.xticks(range(-6, 9, 3), xlabels)

        # add label to first column and last row
        for i in [0, 1, 2]:
            axes[i, 0].set_ylabel("Vel. [m/s]", fontsize=9)
        for i in range(4):
            axes[2, i].set_xlabel("MLT")
            #axes[i].set_xlabel("Solar Local Time")

        # add legend
        axes[1,3].legend(loc="upper right", bbox_to_anchor=(1.33, 1.02),
                         fontsize=7, frameon=True)

        # save the fig
        if center_at_zero_mlt:
            fig_name = "kp_dependency_" + veldir + "_vel_vs_ltm_c0" +\
                        "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1])

        else:
            fig_name = "kp_dependency_" + veldir + "_vel_vs_ltm" +\
                        "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1])

        #plt.figtext(0.5, 0.95, "Stable IMF Interval = " + str(before_mins+del_tm) + " mins",
        #            ha="center", fontsize=15)
        fig.savefig(fig_dir + fig_name + ".png", dpi=300, bbox_inches="tight")
        #fig.savefig(fig_dir + fig_name + ".pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)
            
    # Plot several bins of data in a single panel for a single LAT
    if single_lat:
	del_lat=1
	lat_range=[52, 61]
	#glatc_list = np.arange(lat_range[1]-0.5, lat_range[0]-0.5, -del_lat)
	glatc_list = np.arange(lat_range[0]+0.5, lat_range[1]+0.5, del_lat)

	for season in seasons:
	    if center_at_zero_mlt:
		fig_name = "single_lat_" + season + "_" + veldir + "_vel_vs_ltm_c0"

	    else:
		fig_name = "single_lat_" + season + "_" + veldir + "_vel_vs_ltm"

	    # create subplots
	    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,6), 
				     sharex=True, sharey=True)
	    axes = [ax for l in axes for ax in l]
	    fig.subplots_adjust(hspace=0.4)

	    # fetches the data from db 
	    data_dict_list = []
	    for i, kp_txt in enumerate(kp_text_list):
		input_table = "master_cosfit_" + rads_txt + kp_txt

		data_dict = fetch_data(input_table, lat_range=lat_range,
			    nvel_min=nvel_min, season=season,
			    config_filename="../mysql_dbconfig_files/config.ini",
			    section="midlat", db_name=db_name, ftype=ftype,
			    coords=coords, sqrt_weighting=sqrt_weighting)
		data_dict_list.append(data_dict)

		#color_list = ['darkblue', 'b', 'dodgerblue', 'c', 'g', 'orange', 'r']
		color_list = ['k', 'b', 'g', 'r', 'gray']
		for j, latc in enumerate(glatc_list):
		    ax = axes[j]
		    # plot the flow vector components for each latitude
		    if veldir == "all" :
                        title = "Velocity Magnitude, " + season[0].upper()+season[1:]
		    else:
			title = veldir[0].upper()+veldir[1:] + " Vel, " +\
				season[0].upper()+season[1:] +\
				", MLAT=" + str(latc)
		    vel_vs_lt(ax, data_dict_list[i], veldir=veldir,
			      center_at_zero_mlt=center_at_zero_mlt,
			      glatc_list=[latc], title=title, add_err_bar=add_err_bar,
			      color_list=[color_list[i]], marker_size=2)

		    # change the font
		    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
				 ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(9)
		    ax.legend().set_visible(False)

		    # remove labels
		    for ax in axes:
			ax.set_xlabel("")
			ax.set_ylabel("")

	    # add label to last row
	    for i in range(6,9):
		axes[i].set_xlabel("MLT")
		#axes[i].set_xlabel("Solar Local Time")
		axes[i].xaxis.set_major_locator(MultipleLocator(base=3))

	    # add legend
	    #axes[2].legend(bbox_to_anchor=(1.05, 1.00), fontsize=6)
	    lg = axes[2].legend()
            legend_txt = [kp_text_dict[x][1:] for x in kp_text_list]
	    txts = lg.get_texts()
	    for i in range(len(legend_txt)):
		txts[i].set_text(legend_txt[i])
		txts[i].set_fontsize(9)
	    lg.set_bbox_to_anchor((1.02, 0.93))

	    # save the fig
	    fig.savefig(fig_dir + fig_name + ".png", dpi=300)


def by_imf_clock_angle(single_imf_bin=True, single_lat=False):


    # input parameters
    nvel_min=100
    del_lat=1
    lat_range=[52, 59]
    glatc_list = np.arange(lat_range[1]-0.5, lat_range[0]-0.5, -del_lat)
    if len(glatc_list) == 0:
        glatc_list = np.array([lat_range[0]]+0.5)
    
    ftype = "fitacf"
    coords = "mlt"
    seasons = ["winter"]
    #seasons = ["winter", "equinox", "summer"]
    #veldir="zonal"
    veldir="meridional"
    xlim=[-6, 6]
    ylim=[-80, 30]

    center_at_zero_mlt=True
    sqrt_weighting = True
    add_err_bar = False

    years_txt = ""
    rads_txt = "six_rads"
    kp_text = "_kp_00_to_23"
    db_name = "master_" + coords + "_" + ftype + "_binned_by_imf_clock_angle"

    # Create clock angle bins
    sector_center_dist = 45
    sector_width = 40
    # set bins for all clock angle ranges
    imf_bins = [[x-sector_width/2, x+sector_width/2] for x in np.arange(0, 360, sector_center_dist)]
    #imf_bins = [[-30, 30] for x in imf_bins]
    # Determines how to place the imf_bins into panels,
    # NOTE: must match with imf_bins
    ax_idxs = [1, 2, 5, 8, 7, 6, 3, 0]

#    # set bins for IMF clock angle near 90 or 270
#    sector_centers = [80 - sector_width/2, 100 + sector_width/2,
#                      260 - sector_width/2, 280 + sector_width/2]
#    imf_bins = []
#    for ctr in sector_centers:
#        imf_bins.append([ctr - sector_width/2, ctr + sector_width/2])

#    #bins_txt = ["Bz+", "By+", "Bz-", "By-"]
#    bins_txt = ["By+, Bz+", "By+, Bz-", "By-, Bz-", "By-, Bz+"]
    bins_txt = ["Bz+", "By+/Bz+", "By+", "By+/Bz-",
                "Bz-", "By-/Bz-", "By-", "By-/Bz+"]
    tmp_txt = ""

    # Set IMF stability conditions
    bvec_max = 0.95
    before_mins=50
    after_mins=0
    del_tm=10

    fig_dir = "./plots/velcomp_vs_time/kp_l_3/data_in_mlt/binned_by_imf_clock_angle/"
    # Plot one bin of data in a single panel for several LATs
    if single_imf_bin:
	for j, season in enumerate(seasons):
	    # create subplots
	    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,6), 
				     sharex=True, sharey=True)
	    axes = [ax for l in axes for ax in l]
	    fig.subplots_adjust(hspace=0.3)
            if len(imf_bins) == 1:
                axes = [axes]

	    # fetches the data from db 
	    data_dict_list = []
	    for i, imf_bin in enumerate(imf_bins):
                ax_idx = ax_idxs[i]
		ax = axes[ax_idx]
		input_table = "master_fit_" + rads_txt + kp_text + \
			       "_b" + str((imf_bin[0]%360)) + "_b" + str(imf_bin[1]%360) +\
			       "_bfr" + str(before_mins) +\
			       "_aftr" +  str(after_mins) +\
			       "_bvec" + str(bvec_max).split('.')[-1]

		data_dict = fetch_data(input_table, lat_range=lat_range,
                                       nvel_min=nvel_min, season=season,
                                       config_filename="../mysql_dbconfig_files/config.ini",
                                       section="midlat", db_name=db_name, ftype=ftype,
                                       coords=coords, sqrt_weighting=sqrt_weighting)
		data_dict_list.append(data_dict)

		# Plot the flow vector components for each imf bin
		if veldir == "all" :
		    title = "Velocity Magnitude, " + season[0].upper()+season[1:] + r", Kp $\leq$ 2+"
		else:
		    title = veldir[0].upper()+veldir[1:] + " Vel, " +\
			    season[0].upper()+season[1:] + r", Kp $\leq$ 2+, " +\
			    bins_txt[i]
		vel_vs_lt(ax, data_dict_list[i], veldir=veldir,
			  center_at_zero_mlt=center_at_zero_mlt,
			  glatc_list=glatc_list, title=title, add_err_bar=add_err_bar,
			  color_list=None, marker_size=3, xlim=xlim, ylim=ylim)

		# change the font
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
			     ax.get_xticklabels() + ax.get_yticklabels()):
		    item.set_fontsize(9)
		ax.legend().set_visible(False)

            # Plot the center axis for IMF clock angle
            plot_center_axis(axes[4], sector_center_dist=sector_center_dist,
                             sector_width=sector_width, xlim=xlim, ylim=ylim)


            # Set axis labels
	    # Add label to first column and last row
	    for i in [0, 3, 6]:
		axes[i].set_ylabel("MLAT [degree]", fontsize=9)
	    for i in range(6,9):
		axes[i].set_xlabel("MLT", fontsize=9)
   
                # Set x-axis tick labels
            if center_at_zero_mlt:
                xlabels = [item.get_text() for item in axes[-1].get_xticklabels()]
                xlabels = [str(x) for x in range(18, 24, 3) + range(0, 9, 3)]
                plt.xticks(range(-6, 9, 3), xlabels)
	
	    # add legend
	    axes[3].legend(bbox_to_anchor=(1.02, 0.92), fontsize=7)

	    # save the fig
	    if center_at_zero_mlt:
		fig_name = tmp_txt + season + "_" + veldir + "_vel_vs_ltm_c0" +\
			    "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1]) +\
			    "_bfr" + str(before_mins) +\
			    "_aftr" +  str(after_mins) +\
			    "_bvec" + str(bvec_max).split('.')[-1]
	    else:
		fig_name = tmp_txt + season + "_" + veldir + "_vel_vs_ltm" +\
			    "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1]) +\
			    "_bfr" + str(before_mins) +\
			    "_aftr" +  str(after_mins) +\
			    "_bvec" + str(bvec_max).split('.')[-1]
            #plt.figtext(0.5, 0.95, "Stable IMF Interval = " + str(before_mins+del_tm) + " mins",
            #            ha="center", fontsize=15)
	    fig.savefig(fig_dir + fig_name + ".png", dpi=300, bbox_inches="tight")
	    #fig.savefig(fig_dir + fig_name + ".pdf", format="pdf", bbox_inches="tight")
            plt.close(fig)

    # Plot several bins of data in a single panel for a single LAT
    if single_lat:
        # NOTE: May need some modifitation
	del_lat=1
	lat_range=[52, 61]
	#glatc_list = np.arange(lat_range[1]-0.5, lat_range[0]-0.5, -del_lat)
	glatc_list = np.arange(lat_range[0]+0.5, lat_range[1]+0.5, del_lat)

	for season in seasons:
	    if center_at_zero_mlt:
		fig_name = "single_lat_" + tmp_txt + season + "_" + veldir + "_vel_vs_ltm_c0" +\
			    "_bfr" + str(before_mins) +\
			    "_aftr" +  str(after_mins) +\
			    "_bvec" + str(bvec_max).split('.')[-1]

	    else:
		fig_name = "single_lat_" + tmp_txt + season + "_" + veldir + "_vel_vs_ltm" +\
			    "_bfr" + str(before_mins) +\
			    "_aftr" +  str(after_mins) +\
			    "_bvec" + str(bvec_max).split('.')[-1]

	    # create subplots
	    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,6), 
				     sharex=True, sharey=True)
	    axes = [ax for l in axes for ax in l]
	    fig.subplots_adjust(hspace=0.4)

	    # fetches the data from db 
	    data_dict_list = []
	    for i, imf_bin in enumerate(imf_bins):
		input_table = "master_fit_" + rads_txt + kp_text + \
			       "_b" + str((imf_bin[0]%360)) + "_b" + str(imf_bin[1]%360) +\
			       "_bfr" + str(before_mins) +\
			       "_aftr" +  str(after_mins) +\
			       "_bvec" + str(bvec_max).split('.')[-1]

		data_dict = fetch_data(input_table, lat_range=lat_range,
			    nvel_min=nvel_min, season=season,
			    config_filename="../mysql_dbconfig_files/config.ini",
			    section="midlat", db_name=db_name, ftype=ftype,
			    coords=coords, sqrt_weighting=sqrt_weighting)
		data_dict_list.append(data_dict)

		#color_list = ['darkblue', 'b', 'dodgerblue', 'c', 'g', 'orange', 'r']
		#color_list = ['k', 'b', 'g', 'r']
		color_list = ['k', 'r']
		for j, latc in enumerate(glatc_list):
		    ax = axes[j]
		    # plot the flow vector components for each latitude
		    if veldir == "all" :
			title = "Velocity Magnitude, " + season[0].upper()+season[1:] + r", Kp $\leq$ 2+"
		    else:
			title = veldir[0].upper()+veldir[1:] + " Vel, " +\
				season[0].upper()+season[1:] + r", Kp $\leq$ 2+, " +\
				", MLAT=" + str(latc)
		    vel_vs_lt(ax, data_dict_list[i], veldir=veldir,
			      center_at_zero_mlt=center_at_zero_mlt,
			      glatc_list=[latc], title=title, add_err_bar=add_err_bar,
			      color_list=[color_list[i]], marker_size=2)

		    # change the font
		    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
				 ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(9)
		    ax.legend().set_visible(False)

		    # remove labels
		    for ax in axes:
			ax.set_xlabel("")
			ax.set_ylabel("")

	    # add label to last row
	    for i in range(6,9):
		axes[i].set_xlabel("MLT")
		#axes[i].set_xlabel("Solar Local Time")
		axes[i].xaxis.set_major_locator(MultipleLocator(base=3))

	    # add legend
	    #axes[2].legend(bbox_to_anchor=(1.05, 1.00), fontsize=6)
	    lg = axes[2].legend()
	    legend_txt = bins_txt
	    txts = lg.get_texts()
	    for i in range(len(bins_txt)):
		txts[i].set_text(legend_txt[i])
		txts[i].set_fontsize(9)
	    lg.set_bbox_to_anchor((1.02, 0.93))

	    # save the fig
            plt.figtext(0.5, 0.95, "Stable IMF Interval = " + str(before_mins+del_tm) + " mins",
                        ha="center", fontsize=15)
	    fig.savefig(fig_dir + fig_name + ".png", dpi=300)

    return

def plot_center_axis(ax, sector_center_dist=45, sector_width=40,
                     xlim=[-6, 6], ylim=[-80, 30]):

    import numpy as np

    # Plot arrows
    height = ylim[1] - ylim[0]
    width = xlim[1] - xlim[0]
    #axis_ratio = 1. * height/width
    axis_ratio = 1. * height/width
    aspect_ratio = 0.5
    arrow_len = 0.3 * width    # Arrow length along x-axis
    x1 = xlim[0] + width/2.
    y1 = ylim[0] + height/2.
    imf_bins = [[x-sector_width/2, x+sector_width/2] for x in np.arange(0, 360, sector_center_dist)]
    for i, imf_bin in enumerate(imf_bins):
        sector_center = np.mean(imf_bin)
        len_x = aspect_ratio * arrow_len * np.sin(np.deg2rad(sector_center))
        len_y =  axis_ratio * arrow_len * np.cos(np.deg2rad(sector_center))
        ax.arrow(x1, y1, len_x, len_y, head_width=0.00*arrow_len,
                 head_length=0.0*arrow_len, fc='k', ec='k')

    # Add x-y axis names
    xy_By = (x1 + 1.15*aspect_ratio*arrow_len, y1)
    xy_Bz = (x1, y1 + 1.15*axis_ratio*arrow_len)
    ax.annotate("By+", xy=xy_By, ha="left", va="center")
    ax.annotate("Bz+", xy=xy_Bz, ha="center", va="bottom")

    # Set title
    ax.set_title("IMF Clock Angle", fontsize="medium")

    # remove tikcs and frames
    ax.tick_params(axis='both', which='both', bottom='off', top='off',
                   left="off", right="off", labelbottom='off', labelleft='off')
    ax.axis("off")

    return


if __name__ == "__main__":
    #by_season()
    #six_rads_by_year()
    #by_pairs_of_radars()
    by_kp(single_kp=True, single_lat=False)
    #by_imf_clock_angle(single_imf_bin=True, single_lat=False)
