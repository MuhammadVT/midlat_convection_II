import matplotlib
matplotlib.use('Agg')

def num_plot(ax, data_dict, cmap, bounds,
             lat_min=50, title="xxx"):
    
    """ plots the flow vectors in MLT coords """

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from matplotlib.collections import PolyCollection,LineCollection
    import numpy as np

    from convection import pol2cart 

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

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
        c = plt.Circle((0, 0), radius=r, fill=False, linewidth=.5)
        ax.add_patch(c)

    # plot the longitudinal lines
    for l in np.deg2rad(np.array([210, 240, 270, 300, 330])):
        x1, y1 = pol2cart(l, 10) 
        x2, y2 = pol2cart(l, 50) 
        ax.plot([x1, x2], [y1, y2], 'k', linewidth=.5)

    # plot the velocity locations
    x1, y1 = pol2cart(np.deg2rad(data_dict['glonc']-90), 90-data_dict['glatc'])

    intensities = []
    vel_count = data_dict['vel_count']
    
    #save the param to use as a color scale
    intensities.extend(np.abs(vel_count))

    #do the actual overlay
    ccoll = ax.scatter(x1, y1,
                    s=0.7,zorder=10,marker='o', c=np.abs(np.array(intensities)),
                    linewidths=.5, edgecolors='face'
                    ,cmap=cmap,norm=norm)

    # add labels
    ax.set_title(title, fontsize="small")
    # add latitudinal labels
    fnts = 'x-small'
    ax.annotate("80", xy=(0, -10), ha="left", va="bottom", fontsize=fnts)
    ax.annotate("60", xy=(0, -30), ha="left", va="bottom", fontsize=fnts)
    # add mlt labels
    ax.annotate("0", xy=(0, -rmax), ha="center", va="top", fontsize=fnts)
    ax.annotate("6", xy=(rmax, 0), ha="left", va="center", fontsize=fnts)
    ax.annotate("18", xy=(-rmax, 0), ha="right", va="center", fontsize=fnts)

    return  ccoll

def add_cbar(fig, coll, bounds, label="Number of Measurements", cax=None):

    # add color bar
    if cax:
        cbar=fig.colorbar(coll, cax=cax, orientation="vertical",
                          boundaries=bounds, drawedges=False) 
    else:
        cbar=fig.colorbar(coll, orientation="vertical", shrink=.65,
                          boundaries=bounds, drawedges=False) 


    #define the colorbar labels
    l = []
    for i in range(0,len(bounds)):
        if i == len(bounds)-1:
            l.append(' ')
            continue
        l.append(str(int(bounds[i])))
    cbar.ax.set_yticklabels(l)
    #cbar.ax.tick_params(axis='y',direction='out')
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    return

def by_month():

    import datetime as dt
    import matplotlib.pyplot as plt
    from convection_by_month import fetch_data
    import matplotlib as mpl
    import calendar

    # input parameters
    nvel_min=100
    lat_range=[52, 60]
    lat_min = 50

    ftype = "fitacf"
    coords = "mlt"
    sqrt_weighting = True
    rads_txt = "six_rads"
    input_table = "master_cosfit_" + rads_txt + "_kp_00_to_23_by_month"

    #months = range(1, 13)
    months = [11, 12, 1, 2, 3, 4, 9, 10, 5, 6, 7, 8]

   
    # build a custom color map and bounds
    color_list = ['purple', 'b', 'dodgerblue', 'c', 'g', 'y', 'orange', 'r']
    cmap = mpl.colors.ListedColormap(color_list)
    bounds = range(0, 4000, 500)
    #bounds = range(0, 8000, 1000)
    #bounds = range(100, 800, 100)
    bounds[0] = nvel_min
    bounds.append(20000)

    fig_dir = "./plots/num_measurement_points_by_month/kp_l_3/data_in_mlt/"
    fig_name = rads_txt + "_monthly_v2_num_measurement_points"

    # create subplots
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10,6))
    axes = [ax for l in axes for ax in l]
    fig.subplots_adjust(hspace=-0.5)

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
        title = "Number of Measurements, " + calendar.month_name[month][:3] + r", Kp $\leq$ 2+"
        coll = num_plot(ax, data_dict, cmap, bounds,
                        lat_min=lat_min, title=title)

        # change the font
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(6)

    # add colorbar
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.93, 0.35, 0.01, 0.3])
    add_cbar(fig, coll, bounds, cax=cbar_ax,
	     label="Number of Measurements")

    # save the fig
    fig.savefig(fig_dir + fig_name + ".png", dpi=300)
    #fig.savefig(fig_dir + fig_name + ".pdf", format="pdf")
#    plt.show()

    return

if __name__ == "__main__":
    by_month()
