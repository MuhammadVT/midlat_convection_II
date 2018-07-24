import matplotlib
matplotlib.use('Agg')

def vel_vs_lt(ax, data_dict, veldir="zonal", center_at_zero_mlt=True,
               glatc_list=None, title="xxx", add_err_bar=False):
    
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
    from matplotlib.ticker import MultipleLocator

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
                marker='o', s=0.3, linewidths=.5, edgecolors='face', label=str(mlat))

        if add_err_bar:
            ax.errorbar(vel_mlt_jj, vel_comp_jj, yerr=vel_comp_err_jj, mfc=color_list[jj],
                    #marker='o', s=3, linewidths=.5, edgecolors='face', label=str(int(mlat)))
                    fmt='o', ms=2, elinewidth=.5, mec=color_list[jj], ecolor="k")

    # add text
    ax.set_title(title, fontsize="small")

    # add zero-line
    if veldir != "all":
        ax.axhline(y=0, color='k', linewidth=0.7)

    # set axis limits
    if center_at_zero_mlt:
        ax.set_xlim([-6, 6])
        ax.xaxis.set_major_locator(MultipleLocator(base=3))
        # add legend
        #ax.legend(bbox_to_anchor=(0.98, 0.90), fontsize=7)
        #ax.legend(loc="center right", fontsize=6)
    else:
        ax.set_xlim([0, 24])
        # add legend
        #ax.legend(loc='center right', fontsize=8)
        #ax.legend(bbox_to_anchor=(0.65, 0.96), fontsize=8)

    if veldir == "all":
        ax.set_ylim([0, 60])
    else:
        ax.set_ylim([-75, 25])
    
    # axis labels
    ax.set_ylabel("Vel [m/s]")

    return

def main():

    import datetime as dt
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from convection_by_month import fetch_data
    import numpy as np
    import calendar

    # input parameters
    nvel_min=100
    del_lat=1
    #lat_range=[58, 65]
    lat_range=[52, 59]
    glatc_list = np.arange(lat_range[1]-0.5, lat_range[0]-0.5, -del_lat)

    if len(glatc_list) == 0:
        glatc_list = np.array([lat_range[0]]+0.5)
    
    #add_err_bar = True
    add_err_bar = False
    ftype = "fitacf"
    coords = "mlt"
    sqrt_weighting = True

    #veldir="all"
    #veldir="zonal"
    veldir="meridional"
    center_at_zero_mlt=True
    #center_at_zero_mlt=False

    rads_txt = "six_rads"
    month_txt = "by_month"
    #month_txt = "by_pseudo_month"
    input_table = "master_cosfit_" + rads_txt + "_kp_00_to_23_" + month_txt

    months = range(1, 13)
    #months = [11, 12, 1, 2, 3, 4, 9, 10, 5, 6, 7, 8]

    fig_dir = "./plots/velcomp_vs_time_by_month/kp_l_3/data_in_mlt/"
    if center_at_zero_mlt:
        fig_name = rads_txt + "_" + rads_txt + "_"+ veldir + "_vel_vs_ltm_c0" +\
                   "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1])
    else:
        fig_name = rads_txt + "_" + rads_txt + "_" + veldir+ "_vel_vs_ltm" +\
                   "_lat" + str(lat_range[0]) + "_to_lat" + str(lat_range[1])

    # create subplots
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10,6),
                             sharex=True, sharey=True)
    axes = [ax for l in axes for ax in l]
    fig.subplots_adjust(hspace=0.3)

    if len(months) == 1:
        axes = [axes]

    for i, month in enumerate(months):
        ax = axes[i]
        # fetches the data from db 
        data_dict = fetch_data(input_table, lat_range=lat_range,
                    nvel_min=nvel_min, month=month,
                    config_filename="../mysql_dbconfig_files/config.ini",
                    section="midlat", db_name=None, ftype=ftype,
                    coords=coords, sqrt_weighting=sqrt_weighting)

        # plot the flow vector components
        if veldir == "all" :
            title = "Velocity Magnitude, " + calendar.month_name[month][:3] + r", Kp $\leq$ 2+"
        else:
            title = veldir[0].upper()+veldir[1:] + " Flow, " +\
                    calendar.month_name[month][:3] + r", Kp $\leq$ 2+"
        vel_vs_lt(ax, data_dict, veldir=veldir, center_at_zero_mlt=center_at_zero_mlt,
                glatc_list=glatc_list, title=title, add_err_bar=add_err_bar)

        # change the font
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(7)

    # remove labels
    for ax in axes:
	ax.set_xlabel("")
	ax.set_ylabel("")

    # add label to first column and last row
    for i in [0, 4, 8]:
        axes[i].set_ylabel("Vel. [m/s]", fontsize=9)
    for i in range(8,12):
	axes[i].set_xlabel("MLT")

    # set axis label
    if center_at_zero_mlt:
        xlabels = [item.get_text() for item in axes[-1].get_xticklabels()]
        xlabels = [str(x) for x in range(18, 24, 3) + range(0, 9, 3)]
        plt.xticks(range(-6, 9, 3), xlabels)

    # add legend
    #axes[7].legend(bbox_to_anchor=(1.05, 1.00), fontsize=8, frameon=True)
    axes[4].legend(loc="lower left", fontsize=4, frameon=True)

    # save the fig
    fig.savefig(fig_dir + fig_name + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
