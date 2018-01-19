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
               color_list=None, marker_size=2, marker_type="o"):
    
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
        #ax.set_xlim([-12, 12])
        ax.set_xlim([-6, 6])
        # add legend
        ax.legend(bbox_to_anchor=(1.01, 0.91), fontsize=6)
        #ax.legend(loc="center right", fontsize=6)
    else:
        ax.set_xlim([0, 24])
        # add legend
        #ax.legend(loc='center right', fontsize=8)
        ax.legend(bbox_to_anchor=(0.65, 0.96), fontsize=8)

    if veldir == "all":
        ax.set_ylim([0, 60])
    else:
        #ax.set_ylim([-65, 25])
        ax.set_ylim([-70, 30])
    
    # axis labels
    ax.set_ylabel("Vel [m/s]")

    return

def by_season():

    # input parameters
    nvel_min=300
    #del_lat=3
    del_lat=1
    lat_range=[58, 65]
    #lat_range=[52, 59]

#    nvel_min=100
#    del_lat=1
#    #lat_range=[58, 65]
#    lat_range=[53, 60]


#    nvel_min=100
#    del_lat=1
#    #lat_range=[58, 65]
#    lat_range=[42, 49]


    glatc_list = np.arange(lat_range[1]-0.5, lat_range[0]-0.5, -del_lat)
    #lat_range=[40, 60]
    #glatc_list = np.array([41.5, 50.5])
    #print glatc_list
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

    seasons = ["winter", "summer", "equinox"]
    #seasons = ["winter"]

    sqrt_weighting = True
    rads_txt = "six_rads"
    #rads_txt = "ade_adw"
    #rads_txt = "hok_hkw"

    years = [2011, 2012]
    years_txt = "_years_" + "_".join([str(x) for x in years])
    #years_txt = ""


    #input_table = "master_cosfit_hok_hkw_kp_00_to_23"
    #input_table = "master_cosfit_hok_hkw_kp_00_to_23_azbin_nvel_min_5"
    input_table = "master_cosfit_" + rads_txt + "_kp_00_to_23" + years_txt

    fig_dir = "./plots/velcomp_vs_time/kp_l_3/data_in_mlt/"
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
            title = "Velocity Magnitude, " + season[0].upper()+season[1:] + r", Kp $\leq$ 2+"
        else:
            title = veldir[0].upper()+veldir[1:] + " Velocities, " +\
                    season[0].upper()+season[1:] + r", Kp $\leq$ 2+"
        vel_vs_lt(axes[i], data_dict, veldir=veldir, center_at_zero_mlt=center_at_zero_mlt,
                glatc_list=glatc_list, title=title, add_err_bar=add_err_bar)

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


def by_imf_clock_angle(single_imf_bin=True, single_lat=True):


    # input parameters
    nvel_min=100
    del_lat=1
    lat_range=[53, 59]
    glatc_list = np.arange(lat_range[1]-0.5, lat_range[0]-0.5, -del_lat)
    if len(glatc_list) == 0:
        glatc_list = np.array([lat_range[0]]+0.5)
    
    add_err_bar = False
    ftype = "fitacf"
    coords = "mlt"

    veldir="zonal"
    #veldir="meridional"
    center_at_zero_mlt=True
    sqrt_weighting = True

    years_txt = ""
    rads_txt = "six_rads"
    kp_text = "_kp_00_to_23_"

    #seasons = ["winter", "equinox", "summer"]
    seasons = ["winter"]
    db_name = "master_" + coords + "_" + ftype + "_binned_by_imf_clock_angle"

    # set the imf bins
    sector_width = 60
    sector_center_dist = 90
    imf_bins = [[x-sector_width/2, x+sector_width/2] for x in np.arange(0, 360, sector_center_dist)]
    bins_txt = ["Bz+", "By+", "Bz-", "By-"]

    bvec_max = 0.95
    before_mins=20
    after_mins=10
    del_tm=10

    fig_dir = "./plots/velcomp_vs_time/kp_l_3/data_in_mlt/binned_by_imf_clock_angle/"
    if single_imf_bin:
	for season in seasons:
	    if center_at_zero_mlt:
		fig_name = "single_imf_bin_" + season + "_" + veldir + "_vel_vs_ltm_c0" +\
			    "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1]) +\
			    "_bfr" + str(before_mins) +\
			    "_aftr" +  str(after_mins) +\
			    "_bvec" + str(bvec_max).split('.')[-1]

	    else:
		fig_name = "single_imf_bin_" + season + "_" + veldir + "_vel_vs_ltm" +\
			    "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1]) +\
			    "_bfr" + str(before_mins) +\
			    "_aftr" +  str(after_mins) +\
			    "_bvec" + str(bvec_max).split('.')[-1]

	    # create subplots
	    fig, axes = plt.subplots(nrows=2, ncols=len(imf_bins)/2, figsize=(12,6), 
				     sharex=True, sharey=True)
	    axes = [ax for l in axes for ax in l]
	    fig.subplots_adjust(hspace=0.4)

	    # fetches the data from db 
	    data_dict_list = []
	    for i, imf_bin in enumerate(imf_bins):
		input_table = "master_fit_" + rads_txt + kp_text + \
			       "b" + str((imf_bin[0]%360)) + "_b" + str(imf_bin[1]%360) +\
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
		#markers = ['o', '+', '*', '.']
		ax = axes[i]
		if veldir == "all" :
		    title = "Velocity Magnitude, " + season[0].upper()+season[1:] + r", Kp $\leq$ 2+"
		else:
		    title = veldir[0].upper()+veldir[1:] + " Vel, " +\
			    season[0].upper()+season[1:] + r", Kp $\leq$ 2+, " +\
			    bins_txt[i]
		vel_vs_lt(ax, data_dict_list[i], veldir=veldir,
			  center_at_zero_mlt=center_at_zero_mlt,
			  glatc_list=glatc_list, title=title, add_err_bar=add_err_bar,
			  color_list=None, marker_size=3)

		# change the font
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
			     ax.get_xticklabels() + ax.get_yticklabels()):
		    item.set_fontsize(14)
		ax.legend().set_visible(False)

	    # set axis label
	    if center_at_zero_mlt:
		xlabels = [item.get_text() for item in axes[-1].get_xticklabels()]
		xlabels = [str(x) for x in range(18, 24, 3) + range(0, 9, 3)]
		plt.xticks(range(-6, 9, 3), xlabels)


	    # save the fig
	    fig.savefig(fig_dir + fig_name + ".png", dpi=300)

    if single_lat:
	del_lat=1
	lat_range=[53, 62]
	#glatc_list = np.arange(lat_range[1]-0.5, lat_range[0]-0.5, -del_lat)
	glatc_list = np.arange(lat_range[0]+0.5, lat_range[1]+0.5, del_lat)

	for season in seasons:
	    if center_at_zero_mlt:
		fig_name = "single_lat_" + season + "_" + veldir + "_vel_vs_ltm_c0" +\
			    "_bfr" + str(before_mins) +\
			    "_aftr" +  str(after_mins) +\
			    "_bvec" + str(bvec_max).split('.')[-1]

	    else:
		fig_name = "single_lat_" + season + "_" + veldir + "_vel_vs_ltm" +\
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
			       "b" + str((imf_bin[0]%360)) + "_b" + str(imf_bin[1]%360) +\
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
		color_list = ['k', 'b', 'g', 'r']
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
	    fig.savefig(fig_dir + fig_name + ".png", dpi=300)

    return

if __name__ == "__main__":
    #by_season()
    #six_rads_by_year()
    #by_pairs_of_radars()
    by_imf_clock_angle(single_imf_bin=True, single_lat=True)
