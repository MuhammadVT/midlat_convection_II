import matplotlib
matplotlib.use('Agg')

def num_plot(ax, data_dict, cmap=None, norm=None,
             lat_min=50, title="xxx"):
    
    """ plots the number of data points in each MLAT-MLT grid cell
    in polar frame
    """

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

    # plot the velocity locations
    x1, y1 = pol2cart(np.deg2rad(data_dict['glonc']-90), 90-data_dict['glatc'])

    intensities = []
    vel_count = data_dict['vel_count']
    
    #save the param to use as a color scale
    intensities.extend(np.abs(vel_count))

    #do the actual overlay
    ccoll = ax.scatter(x1, y1,
                    s=4.0,zorder=10,marker='o', c=np.abs(np.array(intensities)),
                    linewidths=.5, edgecolors='face'
                    ,cmap=cmap,norm=norm)

    # add labels
    ax.set_title(title, fontsize=9)
    # add latitudinal labels
    fnts = 'small'
    ax.annotate("80", xy=(0, -10), ha="left", va="bottom", fontsize=fnts)
    ax.annotate("60", xy=(0, -30), ha="left", va="bottom", fontsize=fnts)

    # add mlt labels
    ax.annotate("0", xy=(0, -rmax), xytext=(0, -rmax-1), ha="center", va="top", fontsize=fnts)
    ax.annotate("6", xy=(rmax, 0), xytext=(rmax+1, 0), ha="left", va="center", fontsize=fnts)
    ax.annotate("18", xy=(-rmax, 0), xytext=(-rmax-1, 0), ha="right", va="center", fontsize=fnts)

    return  ccoll

def num_plot_rect(ax, data_dict, cmap=None, norm=None,
                  lat_min=50, title="xxx"):
    
    """ plots the number of data points in each MLAT-MLT grid cell
    in rectangular frame
    """

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.collections import PolyCollection,LineCollection
    import numpy as np
    from convection import pol2cart 
    from matplotlib.ticker import MultipleLocator

    # x-y aspect ration should be equal
    ax.set_aspect("equal")

    # LON to LT and change the range to [-12, 12]
    x1 = np.array([x/15. for x in data_dict['glonc']])
    # rearragen the time to center at mid-night
    for i, x in enumerate(x1):
        if x > 12:
            x1[i] = x - 24
        else:
            x1[i] = x
    # Change the range of LAT to [0, xxx]
    y1 = data_dict['glatc'] - lat_min

    #save the param to use as a color scale
    intensities = []
    vel_count = data_dict['vel_count']
    intensities.extend(np.abs(vel_count))

    #do the actual overlay
    ccoll = ax.scatter(x1, y1,
                       s=4.0,zorder=10,marker='o', c=np.abs(np.array(intensities)),
                       linewidths=.5, edgecolors='face'
                       ,cmap=cmap,norm=norm)

    # set axis limits
    ax.set_xlim([-6, 6])
    ax.xaxis.set_major_locator(MultipleLocator(base=3))

    # Set title 
    ax.set_title(title, fontsize="small")

    # Set y-axis ticks and limits 
    #ymin = 2    # corresponds to 52
    ymin = int(np.floor(np.min(data_dict['glatc'])))
    ymax = int(np.floor(np.max(data_dict['glatc'])))
    ymin_rel = ymin - lat_min
    ymax_rel = ymax - lat_min + 1
    ax.yaxis.set_major_locator(MultipleLocator(base=1))
    ax.set_ylim([ymin_rel, ymax_rel])
    #ylim_range_rel = range(int(ax.get_ylim()[0]), 1+int(ax.get_ylim()[1]))
    ylim_range_rel = [int(x) for x in ax.get_yticks().tolist()]
    ylim_range = [lat_min + x for x in ylim_range_rel]
    ax.set_yticklabels(ylim_range, fontsize="small")

    return  ccoll


def add_cbar(fig, coll, bounds=None, label="Number of Measurements", cax=None):

    # add color bar
    if cax:
        cbar=fig.colorbar(coll, cax=cax, orientation="vertical",
                          boundaries=bounds, drawedges=False) 
    else:
        cbar=fig.colorbar(coll, orientation="vertical", shrink=.65,
                          boundaries=bounds, drawedges=False) 


    #define the colorbar labels
    if bounds:
        l = []
        for i in range(0,len(bounds)):
            if i == 0 or i == len(bounds)-1:
                l.append(' ')
                continue
            l.append(str(int(bounds[i])))
        cbar.ax.set_yticklabels(l)
    else:
        for i in [0, -1]:
            lbl = cbar.ax.yaxis.get_ticklabels()[i]
            lbl.set_visible(False)

    #cbar.ax.tick_params(axis='y',direction='out')
    cbar.set_label(label)

    return

def by_season():

    import datetime as dt
    import matplotlib.pyplot as plt
    from convection import fetch_data
    import matplotlib as mpl


    # input parameters
    nvel_min=100
    #nvel_min=300
    lat_range=[52, 59]
    lat_min = 50

#    nvel_min=100
#    lat_range=[41, 50]
#    lat_min = 40

    ftype = "fitacf"
    coords = "mlt"
    sqrt_weighting = True
    rads_txt = "six_rads"
    #rads_txt = "cve_cvw"
    #rads_txt = "fhe_fhw"
    #rads_txt = "bks_wal"
    #rads_txt = "ade_adw"
    #rads_txt = "hok_hkw"

    #years = [2015, 2016]
    #years_txt = "_years_" + "_".join([str(x) for x in years])
    years_txt = ""
    kp_text = "_kp_00_to_03"
    kp_text_dict ={"_kp_00_to_03" : r", Kp = 0",
                   "_kp_07_to_13" : r", Kp = 1",
                   "_kp_17_to_23" : r", Kp = 2",
                   "_kp_27_to_33" : r", Kp = 3",
                   "_kp_27_to_43" : r", 3-$\leq$Kp$\leq$4+",
                   "_kp_37_to_90" : r", Kp $\geq$ 4-"}


    #input_table = "master_cosfit_hok_hkw_kp_00_to_23_azbin_nvel_min_5"
    input_table = "master_cosfit_" + rads_txt + kp_text + years_txt

    seasons = ["winter", "summer", "equinox"]

    fig_dir = "./plots/num_measurement_points/" + kp_text[1:] + "/data_in_mlt/"
    fig_name = rads_txt + years_txt + "_seasonal_num_measurement_points"
   
    # create subplots
    fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=(6,8))
    fig.subplots_adjust(hspace=0.3)

    # build a custom color map and bounds
    color_list = ['purple', 'b', 'dodgerblue', 'c', 'g', 'y', 'orange', 'r']
    cmap = mpl.colors.ListedColormap(color_list)
    #bounds = range(0, 2400, 300)
    bounds = range(0, 8000, 1000)
    #bounds = range(100, 800, 100)
    bounds[0] = nvel_min
    bounds.append(20000)

    if len(seasons) == 1:
        axes = [axes]

    for i, season in enumerate(seasons):
        # fetches the data from db 
        data_dict = fetch_data(input_table, lat_range=lat_range,
                    nvel_min=nvel_min, season=season,
                    config_filename="../mysql_dbconfig_files/config.ini",
                    section="midlat", db_name=None, ftype=ftype,
                    coords=coords, sqrt_weighting=sqrt_weighting)

        # plot the flow vectors
        title = "Number of Measurements, " + season[0].upper()+season[1:] + kp_text_dict[kp_text] 
        coll = num_plot(axes[i], data_dict, cmap=cmap, norm=norm,
                        lat_min=lat_min, title=title)

    # add colorbar
    fig.subplots_adjust(right=0.78)
    cbar_ax = fig.add_axes([0.83, 0.25, 0.02, 0.5])
    add_cbar(fig, coll, bounds, cax=cbar_ax, label="Number of Measurements")

    # save the fig
    fig.savefig(fig_dir + fig_name + ".png", dpi=300, bbox_inches="tight")
    #fig.savefig(fig_dir + fig_name + ".pdf", format="pdf")
    plt.close(fig)
#    plt.show()

    return

def by_imf_clock_angle():

    import datetime as dt
    import numpy as np
    import matplotlib.pyplot as plt
    from convection import fetch_data
    import matplotlib as mpl
    from convection_by_imf_clock_angle import plot_center_axis

    # input parameters
    #nvel_min=300
    nvel_min=100
    lat_range=[52, 59]
    #lat_range=[40, 60]
    #lat_min = 43
    lat_min = 50

    #frame_type = "circ"    # options: "rect" or "circ"
    frame_type = "rect"
    #cmap_type = "discrete"    # options: "discrete" or "continuous"
    cmap_type = "continuous"    # options: "discrete" or "continuous"

    ftype = "fitacf"
    #ftype = "fitex"
    coords = "mlt"
    sqrt_weighting = True
    #seasons = ["winter", "summer", "equinox"]
    seasons = ["winter"]

    # Construct DB name
    #years = [2011, 2012]
    #years_txt = "_years_" + "_".join([str(x) for x in years])
    years_txt = ""
    rads_txt = "six_rads"
    kp_text = "_kp_00_to_23"
    db_name = "master_" + coords + "_" + ftype + "_binned_by_imf_clock_angle"
   
    # Make cmap and norm
    if cmap_type == "discrete":
        # cmap and bounds for color bar with discrete colors
        color_list = ['purple', 'b', 'dodgerblue', 'c', 'g', 'y', 'orange', 'r']
        cmap = mpl.colors.ListedColormap(color_list)
        bounds = range(100, 500, 50)
        #bounds[0] = 100
        bounds.append(10000)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if cmap_type == "continuous":
        # cmap and bounds for color bar with continuous colors
        cmap = "jet"
        bounds = None
        vmin=0; vmax=1000
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

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
    tmp_txt = "_" + frame_type
    #tmp_txt = "_IMF_By" 

    # Set IMF stability conditions
    bvec_max = 0.95
    before_mins=50
    after_mins=0
    del_tm=10

    for j, season in enumerate(seasons):
        # create subplots
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,6),
                                 sharex=True, sharey=True)
        axes = [x for subls in axes for x in subls]
        fig.subplots_adjust(hspace=0.3)
        if len(imf_bins) == 1:
            axes = [axes]

        # loop through the bins
        for i, imf_bin in enumerate(imf_bins):
            ax_idx = ax_idxs[i]
            # Construct DB table name
            input_table = "master_fit_" + rads_txt + kp_text + \
                           "_b" + str((imf_bin[0]%360)) + "_b" + str(imf_bin[1]%360) +\
                           "_bfr" + str(before_mins) +\
                           "_aftr" +  str(after_mins) +\
                           "_bvec" + str(bvec_max).split('.')[-1]

            # fetches the data from db 
            data_dict = fetch_data(input_table, lat_range=lat_range,
                        nvel_min=nvel_min, season=season,
                        config_filename="../mysql_dbconfig_files/config.ini",
                        section="midlat", db_name=db_name, ftype=ftype,
                        coords=coords, sqrt_weighting=sqrt_weighting)

            # plot the flow vectors
            title = "# Points, " + season[0].upper()+season[1:] +\
                    r", Kp $\leq$ 2+" + ", IMF " + bins_txt[i]
            if frame_type == "circ":
                coll = num_plot(axes[ax_idx], data_dict, cmap=cmap, norm=norm,
                                lat_min=lat_min, title=title)
            if frame_type == "rect":
                coll = num_plot_rect(axes[ax_idx], data_dict, cmap=cmap, norm=norm,
                                     lat_min=lat_min, title=title)

        # Plot the center axis for IMF clock angle
        plot_center_axis(axes[4], sector_center_dist=sector_center_dist,
                         lat_min=lat_min, lat_range=lat_range,
                         sector_width=sector_width, frame_type=frame_type)

        # Set axis labels
        if frame_type == "rect":
            # add label to first column and last row
            for i in [0, 3, 6]:
                axes[i].set_ylabel("MLAT [degree]", fontsize=9)
            for i in range(6,9):
                axes[i].set_xlabel("MLT", fontsize=9)

            # Set x-axis tick labels
            xlabels = [item.get_text() for item in axes[-1].get_xticklabels()]
            xlabels = [str(x) for x in range(18, 24, 3) + range(0, 9, 3)]
            plt.xticks(range(-6, 9, 3), xlabels)

        # add colorbar
        fig.subplots_adjust(right=0.87)
        cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
        add_cbar(fig, coll, bounds, cax=cbar_ax, label="Number of Measurements")

        # save the fig
        fig_dir = "./plots/num_measurement_points/kp_l_3/data_in_mlt/by_imf_clock_angle/"
        fig_name = season + "_num_measurement_points" + tmp_txt +\
                            "_bfr" + str(before_mins) +\
                            "_aftr" +  str(after_mins) +\
                            "_bvec" + str(bvec_max).split('.')[-1] +\
                            "_IMF_interval_" + str(before_mins+after_mins+10)
        fig.savefig(fig_dir + fig_name + ".png", dpi=300, bbox_inches="tight")
        #fig.savefig(fig_dir + fig_name + ".pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)
        plt.show()

        return

if __name__ == "__main__":
    #by_season()
    by_imf_clock_angle()
