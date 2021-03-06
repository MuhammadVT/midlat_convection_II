import matplotlib
matplotlib.use('Agg')

def cosfit_error(ax, data_dict, cmap=None, norm=None,
                 err_type='Magnitude', lat_min=50, title="xxx"):
    
    """ plots the flow vectors in MLT coords """

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.collections import PolyCollection,LineCollection
    import numpy as np
    from convection import pol2cart 

    # plot the backgroud MLT coords
    rmax = 90 - lat_min
    ax.set_xlim([-rmax, rmax])
    ax.set_ylim([-rmax, 0])
    ax.set_aspect("equal")

    # remove tikcs
    ax.tick_params(axis='both', which='both', bottom='off', top='off',
                   left="off", right="off", labelbottom='off', labelleft='off')

    # plot the latitudinal circles
    for r in [10, 30, 50]:
        c = plt.Circle((0, 0), radius=r, fill=False, linewidth=0.5)
        ax.add_patch(c)

    # plot the longitudinal lines
    for l in np.deg2rad(np.array([210, 240, 270, 300, 330])):
        x1, y1 = pol2cart(l, 10) 
        x2, y2 = pol2cart(l, 50) 
        ax.plot([x1, x2], [y1, y2], 'k', linewidth=0.5)

    # calculate the velocity locations in cartisian coords
    x1, y1 = pol2cart(np.deg2rad(data_dict['glonc']-90), 90-data_dict['glatc'])

    #save the param to use as a color scale
    vel_mag_err_ratio = np.abs(np.array(data_dict['vel_mag_err']) / np.array(data_dict['vel_mag']))
    vel_dir_err_ratio = np.abs(np.array(data_dict['vel_dir_err']) / np.array(data_dict['vel_dir']))
    vel_err_ratio = vel_mag_err_ratio * vel_dir_err_ratio
    # normalize
    #vel_err_ratio = vel_err_ratio / np.max(vel_err_ratio)
    if err_type=="Magnitude":
        intensities = vel_mag_err_ratio.tolist()
    if err_type=="Phase":
        intensities = vel_dir_err_ratio.tolist()
    if err_type=="both":
        intensities = vel_err_ratio.tolist()

    #do the actual overlay
    ccoll = ax.scatter(x1, y1,
                    s=2.5,zorder=10,marker='o', c=np.abs(np.array(intensities)),
                    linewidths=.5, edgecolors='face'
                    ,cmap=cmap,norm=norm)

    # add labels
    ax.set_title(title, fontsize='small')
    # add latitudinal labels
    fnts = "x-small"
    ax.annotate("80", xy=(0, -10), ha="left", va="bottom", fontsize=fnts)
    ax.annotate("60", xy=(0, -30), ha="left", va="bottom", fontsize=fnts)
    # add mlt labels
    ax.annotate("0", xy=(0, -rmax), ha="center", va="top", fontsize=fnts)
    ax.annotate("6", xy=(rmax, 0), ha="left", va="center", fontsize=fnts)
    ax.annotate("18", xy=(-rmax, 0), ha="right", va="center", fontsize=fnts)

    return  ccoll

def main():

    import datetime as dt
    import matplotlib.pyplot as plt
    from convection_by_month import fetch_data
    import matplotlib as mpl
    import numpy as np
    import calendar
    from cosfit_quality import cosfit_error_rect, add_cbar

    # input parameters
    nvel_min=100
    lat_range=[52, 59]
    lat_min = 50

    rads_txt = "six_rads"
    input_table = "master_cosfit_" + rads_txt + "_kp_00_to_23_by_month"

    # choose a fitting error type
    err_type = "Magnitude"
    #err_type = "Phase"
    #err_type = "both"

    frame_type = "rect"    # options: "rect" or "circ"
    #frame_type = "circ"
    cmap_type = "discrete"    # options: "discrete" or "continuous"
    #cmap_type = "continuous"    # options: "discrete" or "continuous"

    ftype = "fitacf"
    coords = "mlt"
    sqrt_weighting = True

    #years = [2015, 2016]
    #years_txt = "_years_" + "_".join([str(x) for x in years])
    years_txt = ""
    tmp_txt = "_" + frame_type

    rads_txt = "six_rads"
    month_txt = "by_month"
    #month_txt = "by_pseudo_month"
    input_table = "master_cosfit_" + rads_txt + "_kp_00_to_23_" + month_txt

    months = range(1, 13)
    #months = [11, 12, 1, 2, 3, 4, 9, 10, 5, 6, 7, 8]

    # Make cmap and norm
    if cmap_type == "discrete":
        # cmap and bounds for color bar with discrete colors
        color_list = ['purple', 'b', 'dodgerblue', 'c', 'g', 'y', 'orange', 'r']
        cmap = mpl.colors.ListedColormap(color_list)
        if err_type == "both":
            bounds = np.arange(0, 0.09, 0.01).tolist()
        else:
            bounds = np.arange(0, 0.9, 0.1).tolist()
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if cmap_type == "continuous":
        # cmap and bounds for color bar with continuous colors
        cmap = "jet"
        bounds = None
        vmin=0.; vmax=1.0
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # create subplots
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12,5))
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

        # plot the flow vectors
        title = "Fitting Quality, " + calendar.month_name[month][:3] + r", Kp $\leq$ 2+"
        if frame_type == "circ":
            coll = cosfit_error(ax, data_dict, cmap=cmap, norm=norm, err_type=err_type,
                                lat_min=lat_min, title=title)
        if frame_type == "rect":
            coll = cosfit_error_rect(ax, data_dict, cmap=cmap, norm=norm, err_type=err_type,
                                     lat_min=lat_min, title=title)
        # change the font
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(6)

    # add colorbar
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.93, 0.25, 0.01, 0.5])
    if err_type == "both":
        cbar_label = "Fitted Vel Magnitude & Phase Error Ratio"
    else:
        #cbar_label = "Fitted Vel " + err_type + " Error Ratio"
        cbar_label = "Fitted Velocity Error Ratio"
    add_cbar(fig, coll, bounds=bounds, cax=cbar_ax, label=cbar_label)

    # save the fig
    fig_dir = "./plots/cosfit_quality_by_month/kp_l_3/data_in_mlt/"
    fig_name = rads_txt + "_monthly_cosfit_quality" + tmp_txt + "_" +\
               str(lat_range[0]) +"_to_lat" + str(lat_range[1])
    fig.savefig(fig_dir + fig_name + ".png", dpi=300, bbox_inches="tight")
    #fig.savefig(fig_dir + fig_name + ".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return

if __name__ == "__main__":
    main()
