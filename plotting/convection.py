import matplotlib
matplotlib.use('Agg')

def fetch_data(input_table, lat_range=[52, 59], nvel_min=250, season="winter",
               config_filename="../mysql_dbconfig_files/config.ini",
               section="midlat", db_name=None, ftype="fitacf",
	       coords="mlt", sqrt_weighting=True, limit_to_night=True):

    """ fetch fitted data from the master db into a dict

    Parameters
    ----------
    input_table : str
        A table name in db_name db
    lat_ragne : list
	The range of latitudes of interest
    nvel_min : int
        minimum requirement for the number of velocity measurements
	in a lat-lon grid cell
    season : str
        season of interest
    config_filename: str
        name and path of the configuration file
    section: str, default to "midlat"
        section of database configuration
    db_name : str, default to None
        Name of the master db
    ftype : str
        SuperDARN file type
    coords : str
        Coordinates in which the binning process took place.
        Default to "mlt, can be "geo" as well.
    sqrt_weighting : bool
        if set to True, the fitted vectors that are produced through weighting the
        number of points within each azimuthal bin will be retrieved. 
        if set to False, the fitted vectors that are produced by equality weighting
        the number of points within each azimuthal bin will be retrieved.

    Return
    ------
    data_dict : dict

    """
    import sqlite3
    import datetime as dt
    import numpy as np 
    from mysql.connector import MySQLConnection
    import sys
    sys.path.append("../")
    from mysql_dbutils.db_config import db_config
    import logging

    # construct a db name
    if db_name is None:
        db_name = "master_" + coords + "_" +ftype

    # read db config info
    config =  db_config(config_filename=config_filename, section=section)
    config_info = config.read_db_config()

    # make a connection to master db
    try:
        conn = MySQLConnection(database=db_name, **config_info)
        cur = conn.cursor(buffered=True)
    except Exception, e:
        logging.error(e, exc_info=True)

    # set input_table name
    if sqrt_weighting:
        input_table = input_table
    else:
        input_table = input_table + "_equal_weighting"

    # formulate column names
    if coords == "mlt":
        col_glatc = "mag_glatc"   # glatc -> gridded latitude center
        col_gltc = "mag_gltc"     # mlt hour in degrees
        col_gazmc = "mag_gazmc"   # gazmc -> gridded azimuthal center
        col_gazmc_count = "mag_gazmc_count"
    if coords == "geo":
        col_glatc = "geo_glatc"
        col_gltc = "geo_gltc"    # local time in degrees
        col_gazmc = "geo_gazmc"
        col_gazmc_count = "geo_gazmc_count"


    # select data from the master_cosfit table for the night side
    if limit_to_night:
        command = "SELECT vel_count, vel_mag, vel_dir, {glatc}, {gltc}, " +\
                  "vel_mag_err, vel_dir_err, season FROM {tb1} " +\
                  "WHERE ({glatc} BETWEEN {lat_min} AND {lat_max}) " +\
                  "AND (({gltc} BETWEEN 269 AND 361) or ({gltc} BETWEEN 0 AND 91)) " +\
                  "AND season = '{season}'"
    else:
        command = "SELECT vel_count, vel_mag, vel_dir, {glatc}, {gltc}, " +\
                  "vel_mag_err, vel_dir_err, season FROM {tb1} " +\
                  "WHERE ({glatc} BETWEEN {lat_min} AND {lat_max}) " +\
                  "AND season = '{season}'"
    command = command.format(tb1=input_table, glatc=col_glatc,
                             gltc = col_gltc, lat_min=lat_range[0],
                             lat_max=lat_range[1], season=season)
    cur.execute(command)
    rws = cur.fetchall()

    data_dict = {}
    # filter the data based on lattitude range. 
    data_dict['vel_count'] = np.array([x[0] for x in rws if x[0] >= nvel_min])
    data_dict['vel_mag'] = np.array([x[1] for x in rws if x[0] >= nvel_min])
    data_dict['vel_dir'] = np.array([x[2] for x in rws if x[0] >= nvel_min])
    data_dict['glatc'] = np.array([x[3] for x in rws if x[0] >= nvel_min])
    data_dict['glonc'] = np.array([x[4] for x in rws if x[0] >= nvel_min])
    data_dict['vel_mag_err'] = np.array([x[5] for x in rws if x[0] >= nvel_min])
    data_dict['vel_dir_err'] = np.array([x[6] for x in rws if x[0] >= nvel_min])
    data_dict['season'] = np.array([x[7] for x in rws])

    # close db connection
    conn.close()

    return data_dict

def cart2pol(x, y):
    import numpy as np
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (phi, rho)

def pol2cart(phi, rho):
    import numpy as np
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def vector_plot(ax, data_dict, cmap, bounds, velscl=1, lat_min=50, title="xxx",
                hemi="north", fake_pole=False):
    
    """ plots the flow vectors in LAT/LT grids in coords

    Parameters
    ----------
    
    """

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.collections import PolyCollection,LineCollection
    import numpy as np

    # build a custom color map
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # set axis limits
    if fake_pole:
        fake_pole_lat = 70
    else:
        fake_pole_lat = 90

    rmax = fake_pole_lat - lat_min
    ax.set_xlim([-rmax, rmax])
    ax.set_ylim([-rmax, 0])
    ax.set_aspect("equal")

    # remove tikcs
    ax.tick_params(axis='both', which='both', bottom='off', top='off',
                   left="off", right="off", labelbottom='off', labelleft='off')

    # plot the latitudinal circles
    for r in [10, 30, 50]:
        c = plt.Circle((0, 0), radius=r, fill=False)
        ax.add_patch(c)

    # plot the longitudinal lines
    for l in np.deg2rad(np.array([210, 240, 270, 300, 330])):
        x1, y1 = pol2cart(l, 10) 
        x2, y2 = pol2cart(l, 50) 
        ax.plot([x1, x2], [y1, y2], 'k')

    x1, y1 = pol2cart(np.deg2rad(data_dict['glonc']-90),
                      fake_pole_lat-np.abs(data_dict['glatc']))

    # add the vector lines
    lines = []
    intensities = []
    vel_mag = data_dict['vel_mag']

    # calculate the angle of the vectors in a tipical x-y axis.
    theta = np.deg2rad(data_dict['glonc'] + 90 - data_dict['vel_dir']) 

    # make the points sparse
    sparse_factor = 2
    x1 = np.array([x1[i] for i in range(len(x1)) if i%sparse_factor==0])
    y1 = np.array([y1[i] for i in range(len(y1)) if i%sparse_factor==0])
    vel_mag = np.array([vel_mag[i] for i in range(len(vel_mag)) if i%sparse_factor==0])
    theta = np.array([theta[i] for i in range(len(theta)) if i%sparse_factor==0])


    x2 = x1+vel_mag/velscl*(-1.0)*np.cos(theta)
    y2 = y1+vel_mag/velscl*(-1.0)*np.sin(theta)
    lines.extend(zip(zip(x1,y1),zip(x2,y2)))

    #save the param to use as a color scale
    intensities.extend(np.abs(vel_mag))

    # plot the velocity locations
    ccoll = ax.scatter(x1, y1,
                        s=1.0,zorder=10,marker='o', c=np.abs(np.array(intensities)),
                        linewidths=.5, edgecolors='face'
                        ,cmap=cmap,norm=norm)
    lcoll = LineCollection(np.array(lines),linewidths=0.5,zorder=12
                        ,cmap=cmap,norm=norm)
    lcoll.set_array(np.abs(np.array(intensities)))
    ccoll.set_array(np.abs(np.array(intensities)))
    ax.add_collection(ccoll)
    ax.add_collection(lcoll)

    # add text
    ax.set_title(title, fontsize='small')
    # add latitudinal labels
    fnts = 'x-small'
    if hemi=="north":
        ax.annotate("80", xy=(0, -10), ha="left", va="bottom", fontsize=fnts)
        ax.annotate("60", xy=(0, -30), ha="left", va="bottom", fontsize=fnts)
    elif hemi=="south":
        ax.annotate("-80", xy=(0, -10), ha="left", va="bottom", fontsize=fnts)
        ax.annotate("-60", xy=(0, -30), ha="left", va="bottom", fontsize=fnts)

    # add mlt labels
    ax.annotate("0", xy=(0, -rmax), ha="center", va="top", fontsize=fnts)
    ax.annotate("6", xy=(rmax, 0), ha="left", va="center", fontsize=fnts)
    ax.annotate("18", xy=(-rmax, 0), ha="right", va="center", fontsize=fnts)

    return lcoll

def add_cbar(fig, coll, bounds, label="Velocity [m/s]", cax=None):

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
        if i == 0 or i == len(bounds)-1:
            l.append(' ')
            continue
        l.append(str(int(bounds[i])))
    cbar.ax.set_yticklabels(l)
    #cbar.ax.tick_params(axis='y',direction='out')
    cbar.set_label(label)

    return


def main():

    import datetime as dt
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # input parameters
    #nvel_min=300
    nvel_min=100
    lat_range=[52, 59]
    lat_min = 50

    # for HOK, HKW radars
#    nvel_min=100
#    lat_range=[42, 49]
#    lat_min = 40

    ftype = "fitacf"
    coords = "mlt"
    sqrt_weighting = True
    #rads_txt = "six_rads"
    #rads_txt = "cve_cvw"
    #rads_txt = "fhe_fhw"
    #rads_txt = "bks_wal"
    rads_txt = "ade_adw"
    #rads_txt = "hok_hkw"

    years = [2013, 2014]
    years_txt = "_years_" + "_".join([str(x) for x in years])
    #years_txt = ""

    #input_table = "master_cosfit_hok_hkw_kp_00_to_23"
    #input_table = "master_cosfit_hok_hkw_kp_00_to_23_azbin_nvel_min_5"
    input_table = "master_cosfit_" + rads_txt + "_kp_00_to_23" + years_txt

    # cmap and bounds for color bar
    color_list = ['purple', 'b', 'c', 'g', 'y', 'r']
    cmap = mpl.colors.ListedColormap(color_list)
    bounds = [0., 8, 17, 25, 33, 42, 10000]

    seasons = ["winter", "summer", "equinox"]

    fig_dir = "./plots/convection/kp_l_3/data_in_mlt/"
    fig_name = rads_txt + years_txt + "_seasonal_convection_lat" + str(lat_range[0]) +\
               "_to_lat" + str(lat_range[1]) + "_nvel_min_" + str(nvel_min)
   
    # create subplots
    fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=(6,8))
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


        # plot the flow vectors
        title = "Velocities, " + season[0].upper()+season[1:] + r", Kp $\leq$ 2+"
        coll = vector_plot(axes[i], data_dict, cmap, bounds, velscl=10,
                           lat_min=lat_min, title=title)

    # add colorbar
    fig.subplots_adjust(right=0.80)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5])
    add_cbar(fig, coll, bounds, cax=cbar_ax, label="Velocity [m/s]")
    # save the fig
    fig.savefig(fig_dir + fig_name + ".png", dpi=300)
    #plt.show()

    return

if __name__ == "__main__":
    main()
