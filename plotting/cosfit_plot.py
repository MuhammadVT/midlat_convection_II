import matplotlib
matplotlib.use('Agg')

def plot_cosfit(ax, latc, ltc, summary_table, cosfit_table, season="winter",
                config_filename="../mysql_dbconfig_files/config.ini",
                section="midlat", db_name=None, ftype="fitacf",
                coords="mlt", sqrt_weighting=True, add_errbar=False):

    """ plots a the cosfit results for a give latc-ltc grid for a giving season"""

    import numpy as np
    import matplotlib.pyplot as plt
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
        cosfit_table = cosfit_table
    else:
        cosfit_table = cosfit_table + "_equal_weighting"

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

    # Find points that best matches the user input latc, ltc.
    command = "SELECT {glatc}, {gltc} FROM {tb} WHERE season='{season}'"
    command = command.format(tb=cosfit_table, glatc=col_glatc,
                             gltc = col_gltc, season=season)
    cur.execute(command)
    rows = cur.fetchall()
    all_lats = np.array([x[0] for x in rows])
    matching_lats_idx = np.where(all_lats==latc)
    all_lts = np.array([x[1] for x in rows])
    possible_lts = all_lts[matching_lats_idx]
    ltc_idx = (np.abs(possible_lts - ltc)).argmin()
    ltc = round(possible_lts[ltc_idx],2)
    
    # Find the AZM and LOS info
    command = "SELECT vel_median, vel_count, {gazmc}, vel_std FROM {tb} " +\
              "WHERE {glatc}={lat} AND {gltc}={lt} AND season='{season}' ORDER BY {gazmc}"
    command = command.format(tb=summary_table, glatc=col_glatc, gltc = col_gltc,
                             gazmc = col_gazmc, lat=latc, lt=ltc, season=season)
    cur.execute(command)
    rows = cur.fetchall()
    median_vel = -np.array([x[0] for x in rows])
    vel_std = np.array([x[3] for x in rows])
    gazmc_vel_count = np.array([x[1] for x in rows])
    weight =  np.sqrt(gazmc_vel_count)
    azm = np.array([x[2] for x in rows])
    azm = [x if x <= 180 else x-360 for x in azm]

    # select the cosine fitting results from db
    command = "SELECT vel_count, vel_mag, vel_mag_err, vel_dir, " + \
              "vel_dir_err FROM {tb} WHERE {glatc}={lat} " +\
              "AND {gltc}={lt} AND season='{season}'"
    command = command.format(tb=cosfit_table, glatc=col_glatc,
                             gltc = col_gltc, lat=latc, lt=ltc,
                             season=season)
    cur.execute(command)
    row = cur.fetchall()[0]
    vel_count = row[0]
    vel_mag = -row[1]
    vel_mag_err = row[2]
    vel_dir = row[3]
    vel_dir_err = row[4]

    # close db connection
    conn.close()

    # plot the LOS data
    ax.scatter(azm, median_vel, marker='o',c='k', s=0.6*weight,
               edgecolors="face", label="LOS Vel.")

    # add error bars to LOS vels
    if add_errbar:
        ax.errorbar(azm, median_vel, yerr=vel_std, capsize=1, mfc='k',
                fmt='o', ms=2, elinewidth=.5, mec='k', ecolor="k")

    # plot the cosfit curve
    #x_fit = np.arange(0, 360, 1)
    x_fit = np.arange(-180, 180, 0.01)
    y_fit = vel_mag * np.cos(np.deg2rad(x_fit) - np.deg2rad(vel_dir))
    ax.plot(x_fit, y_fit, 'y', linewidth=1, label="Fit Line")

    # mark the peak position
    ind_max = np.argmax(y_fit)
    y_max = y_fit[ind_max]
    x_max = x_fit[ind_max]
    fsz = 5
    ax.scatter(x_max, y_max, c='r', edgecolors="face", marker = '*', s = 50, label="Fitted Vel.", zorder=5)
    ax.annotate('vel=' + '{0:.01f}'.format(y_max) , xy = (0.02, 0.88), xycoords='axes fraction',\
       horizontalalignment='left', verticalalignment='bottom', fontsize=fsz) 
    ax.annotate('azm=' + '{0:.01f}'.format(x_max) +'$^\circ$' , xy = (0.015, 0.78), xycoords='axes fraction',\
       horizontalalignment='left', verticalalignment='bottom', fontsize=fsz) 
    
    # fitting error values
#    ax.annotate('vel_std=' + '{0:.01f}'.format(vel_mag_err) , xy = (0.02, 0.74), xycoords='axes fraction',\
#       horizontalalignment='left', verticalalignment='bottom', fontsize=fsz) 
#    ax.annotate('azm_std=' + '{0:.01f}'.format(vel_dir_err) +'$^\circ$' , xy = (0.02, 0.66), xycoords='axes fraction',\
#            horizontalalignment='left', verticalalignment='bottom', fontsize=fsz) 
    
    ax.set_xlim([-180, 180])
    ax.set_ylim([-100, 100])

    # put labels
#    ax.set_title("Velocity Fitting Results, " + season[0].upper()+season[1:] +\
#                 ", MLat = " + str(latc) + ", MLT = " + str(round(ltc/15., 2)))
    ax.set_title(season[0].upper()+season[1:] +\
                 ", MLat = " + str(latc) + ", MLT = " + str(round(ltc/15., 1)))
    ax.set_xlabel("Azimuth [$^\circ$]")
    ax.set_ylabel("Velocity [m/s]")

    return
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import sys
    sys.path.append("../data_preprocessing/")
    from bin_data import grids 
    import numpy as np
    import sqlite3

    #sqrt_weighting=False
    sqrt_weighting=True
    ftype = "fitacf"
    coords = "mlt"
    rads_txt = "six_rads"
    #rads_txt = "ade_adw"
    #rads_txt = "cve_cvw"
    #rads_txt = "fhe_fhw"
    #rads_txt = "bks_wal"
    #rads_txt = "hok_hkw"

    #years = [2011, 2012]
    #years_txt = "_years_" + "_".join([str(x) for x in years])
    years_txt = ""
    #kp_text = "_kp_00_to_23"
    kp_text = "_kp_37_to_90"
    kp_text_dict ={"_kp_00_to_03" : r", Kp = 0",
                   "_kp_00_to_23" : r", Kp$\leq$2+",
                   "_kp_07_to_13" : r", Kp = 1",
                   "_kp_17_to_23" : r", Kp = 2",
                   "_kp_27_to_33" : r", Kp = 3",
                   "_kp_27_to_43" : r", 3-$\leq$Kp$\leq$4+",
                   "_kp_37_to_90" : r", Kp $\geq$ 4-"}


    #input_table = "master_cosfit_hok_hkw_kp_00_to_23_azbin_nvel_min_5"
    summary_table = "master_summary_" + rads_txt + kp_text + years_txt
    cosfit_table = "master_cosfit_" + rads_txt + kp_text + years_txt

    season = "winter"
    #season = "summer"
    #season = "equinox"

    fixed_lat = True
    fixed_lt = True
    # Plot points at a given latitude
    if fixed_lat:
        # points of interest
        #latc, ltc = 46.5, 0
        #latc_list = [x+0.5 for x in range(42, 50)]
        latc_list = [x+0.5 for x in range(52, 60)]
        ltc_list = range(270, 360, 15) + range(0, 90, 15)
        
        # plotting
        for latc in latc_list:
            # create a figure
            fig, axes = plt.subplots(4,3, sharex=True, sharey=True)
            plt.subplots_adjust(hspace=0.4)
            axes = [x for l in axes for x in l]

            fig_dir = "./plots/cosfit_plot/" + kp_text[1:] + "/data_in_mlt/"
            #fig_name = rads_txt + "_" + season + "_cosfit_mlat"+str(latc) + \
            #           "_mlt" + str(round(ltc/15., 2))
            fig_name = rads_txt + years_txt + "_" + season + "_cosfit_mlat"+str(latc)


            for i, ltc in enumerate(ltc_list):
                ax = axes[i]
                plot_cosfit(ax, latc, ltc, summary_table, cosfit_table, season=season,
                            config_filename="../mysql_dbconfig_files/config.ini",
                            section="midlat", db_name=None, ftype=ftype,
                            coords=coords, sqrt_weighting=sqrt_weighting)

                # change the font
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(6)
            for ax in axes:
                ax.set_xlabel("")
                ax.set_ylabel("")
            
            # add legend
            axes[5].legend(bbox_to_anchor=(1, 0.7), fontsize=5, frameon=False)

            # save the plot
	    plt.figtext(0.5, 0.95, kp_text_dict[kp_text][1:], ma="center")
            fig.savefig(fig_dir + fig_name + ".png", dpi=300)

    # Plot points at a given local time
    if fixed_lt:
        # points of interest
        #latc, ltc = 46.5, 0
        #latc_list = [x+0.5 for x in range(42, 54)]
        latc_list = [x+0.5 for x in range(52, 64)]
        #ltc_list = range(270, 360, 15) + range(0, 90, 15)
        ltc_list = range(270, 360, 30) + range(0, 120, 30)
        
        # plotting
        for ltc in ltc_list:
            # create a figure
            fig, axes = plt.subplots(4,3, sharex=True, sharey=True)
            plt.subplots_adjust(hspace=0.4)
            axes = [x for l in axes for x in l]

            fig_dir = "./plots/cosfit_plot/" + kp_text[1:] + "/data_in_mlt/"
            fig_name = rads_txt + years_txt + "_" + season + "_mlt" + str(round(ltc/15., 0))


            for i, latc in enumerate(latc_list):
                ax = axes[i]
                plot_cosfit(ax, latc, ltc, summary_table, cosfit_table, season=season,
                            config_filename="../mysql_dbconfig_files/config.ini",
                            section="midlat", db_name=None, ftype=ftype,
                            coords=coords, sqrt_weighting=sqrt_weighting)

                # change the font
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(6)
            for ax in axes:
                ax.set_xlabel("")
                ax.set_ylabel("")
            
            # add legend
            axes[5].legend(bbox_to_anchor=(1, 0.7), fontsize=5, frameon=False)

            # save the plot
	    plt.figtext(0.5, 0.95, kp_text_dict[kp_text][1:], ma="center")
            fig.savefig(fig_dir + fig_name + ".png", dpi=300)

