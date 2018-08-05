import matplotlib.pyplot as plt
plt.style.use("ggplot")

def read_ssusi_aurora_data(sdate, edate, file_dir,
                           sat_num="f16", hemi="north"):
    """Reads DMSP SSUSI EDR-AUR data type"""
    import netCDF4
    import datetime as dt
    import numpy as np
    import pandas as pd
    import glob

    dtms = []
    mlat = []
    mlt = []
    dtm = sdate
    while dtm <=edate:
        fl_dir = file_dir + sat_num +  "/" + dtm.strftime("%Y%m%d") + "/"
        fnames = glob.glob(fl_dir + "*")
        for file_name in fnames:
            # Convert data format from netCDF4 to pandas DataFrame
            ds = netCDF4.Dataset(file_name)
            magnetic_latitude = ds.variables[hemi.upper() + "_GEOMAGNETIC_LATITUDE"][:]
            magnetic_local_time = ds.variables[hemi.upper() + "_MAGNETIC_LOCAL_TIME"][:]
            ut_seconds =  ds.variables["TIME"][:]
            ut_time = dtm + dt.timedelta(seconds=ut_seconds + 0)
            mlat.append(magnetic_latitude)
            mlt.append(magnetic_local_time)
            dtms.append(ut_time)
        dtm = dtm + dt.timedelta(days=1)

    # Construct a dataframe
    # Drop seconds and milleseconds
    dtms = [x.replace(second=0, microsecond=0) for x in dtms]
    df = pd.DataFrame(data={"mlat":mlat, "mlt":mlt, "hemi":hemi, "sat_num":sat_num},
                      index=dtms)

    # Sort by datetime
    df.sort_index(inplace=True)
    
    return df

def plot_auroral_boundary(ax, df, hemi="north", linestyle="", marker="o",
                          markersize=0.5, mec="k", mfc="k", alpha=1.0):

    import numpy as np

    #mlats = np.array([np.array([float(x) for x in lat.split("_")]) for lat in df.mlat.as_matrix()])
    #mlts = np.array([np.array([float(x) for x in lt.split("_")]) for lt in df.mlt.as_matrix()])
    #mlats = df.mlat
    #mlts = df.mlt
    #mlats = np.array([mlats[1], mlats[2]])
    #mlts = np.array([mlts[1], mlts[2]])

    # Iter through each row
    for i, rw in df.iterrows():
        # Convert (mlats, mlts) to (phi, radius)
        #r = [90-x for arr in mlats for x in arr]
        r = [90-mlat for mlat in rw.mlat]
        # rotate phi by 90 degree to shift the phi=0 line to nightside
        #phi = [np.deg2rad(15*x-90) for arr in mlts for x in arr]
        phi = [np.deg2rad(15*mlt-90) for mlt in rw.mlt]
        ax.plot(phi, r, linestyle="", marker=marker, markersize=markersize,
                mec=mec, mfc=mfc, alpha=alpha)

    # Set MLAT labels
    mlat_boundary = 50
    mlat_sep = 10
    ax.set_rmax(90 - mlat_boundary)
    rgrid_num = (90-mlat_boundary) / mlat_sep
    lst = ['']*(rgrid_num-1)
    if hemi=="north":
        lst.append(str(mlat_boundary)+'$^\circ$')
        ax.set_rgrids(range(mlat_sep, (90-mlat_boundary)+mlat_sep, mlat_sep),
                      angle=-45, ha='left',va='bottom', labels=lst)

    elif hemi=="south":
        lst.append(str(-mlat_boundary)+'$^\circ$')
        ax.set_rgrids(range(mlat_sep, (90-mlat_boundary)+mlat_sep, mlat_sep),
                angle=-90,ha='left',va='top',labels=lst)
    ax.set_thetagrids(range(0, 360, 90), frac=1.07, labels=['06', '12', '18', '00'])

    return

def extract_aurora_boundary(df, mlt=18):

    import numpy as np
    import pandas as pd

    mlats = np.array([np.array([float(x) for x in lat.split("_")]) for lat in df.mlat.as_matrix()])
    mlts = np.array([np.array([float(x) for x in lt.split("_")]) for lt in df.mlt.as_matrix()])
    if mlt == 0:
        idx = [np.argmin(np.min(np.vstack(np.abs((24 - (lt-mlt)), np.abs(lt-mlt)))), axis=0) for lt in mlts]
    else:
        idx = [np.argmin(np.abs(lt-mlt)) for lt in mlts]
    mlat_pnts = [mlats[i][idx[i]] for i in range(len(mlats))]
    mlt_pnts = [mlts[i][idx[i]] for i in range(len(mlts))]
    dfn = pd.DataFrame(data={"mlat":mlat_pnts, "mlt":mlt,
                             "mlt_true":mlt_pnts, "hemi":df.hemi.as_matrix(),
                             "sat_num":df.sat_num.as_matrix()},
                       index=df.index)
    return dfn

def add_au_al(df_mlt):

    import ae
    import pandas as pd
    import datetime as dt

    # read AE data
    stime = df_mlt.index[0]
    etime = df_mlt.index[-1]
    AE_list = ae.readAeWeb(sTime=stime,eTime=etime,res=1)

    # Select items of interest
    dtms = df_mlt.index
    AE_tmp = [x for x in AE_list if x.time in df_mlt.index]
    AU = []
    AL = []
    AE = []
    AE_dtms = []
    for m in AE_tmp:
        AU.append(m.au)
        AL.append(m.al)
        AE.append(m.ae)
        AE_dtms.append(m.time)
    df_tmp = pd.DataFrame(data={"AU":AU, "AL":AL, "AE":AE},
                          index=AE_dtms)

    df_mlt = df_mlt.join(df_tmp, how="left")
   
    return df_mlt

def add_imf(df_ae, time_delay=15):

    import sqlite3
    import pandas as pd
    import numpy as np
    import datetime as dt

    # make a db connection
    dbdir = "../../data/sqlite3/"
    dbname = "gmi_imf.sqlite"
    conn = sqlite3.connect(dbdir + dbname, detect_types = sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    stm = df_ae.index[0] - dt.timedelta(hours=1)
    etm = df_ae.index[-1] + dt.timedelta(hours=1)
    input_table = "IMF"
    command = "SELECT Bx, By, Bz, theta, datetime FROM {tb} " + \
              "WHERE datetime BETWEEN '{stm}' AND '{etm}' "
    command = command.format(tb=input_table, stm=stm, etm=etm)
    cur.execute(command)
    rws = cur.fetchall()
    Bx = [x[0] for x in rws]
    By = [x[1] for x in rws]
    Bz = [x[2] for x in rws]
    theta = [x[3] for x in rws]
    dtms = [x[4] for x in rws]
    df_tmp = pd.DataFrame(data={"Bx":Bx, "By":By, "Bz":Bz, "theta":theta},
                          index=dtms)
   
    df_tmp = df_tmp.shift(time_delay).dropna()
    df_ae = df_ae.join(df_tmp, how="left")

    return df_ae

def prepare_data_for_project(df_all):
        
    import numpy as np

    # Remove mlt=0
    dfn = df_all.loc[df_all.mlt != 0, :]
    
    # Set mlt=18 to 1 and mlt=12 to 0
    dfn.loc[:, "mlt_18"] = (dfn.mlt.as_matrix() == 18).astype(int)

    # Drop NaN
    dfn = dfn.dropna()

    # Remove points whose mlt values are not close to preset mlt
    dfn = dfn.loc[np.abs(dfn.mlt.as_matrix() - dfn.mlt_true.as_matrix()) < 1, :]

    # Drop some columns
    #dfn = dfn.drop(labels=["hemi", "AE", "Bx", "theta", "mlt", "mlt_true"], axis=1)
    dfn = dfn.drop(labels=["hemi", "AE", "mlt", "mlt_true"], axis=1)

    return dfn 

if __name__ == "__main__":

    import datetime as dt
    import pandas as pd

    sdate = dt.datetime(2011,1,1)
    edate = dt.datetime(2011,2,1)

    sat_nums = ["f16", "f16", "f16",
                "f17", "f17", "f17",
                "f18", "f18", "f18"]
    mlts=[18, 18, 18, 6, 6, 6, 0, 0, 0]
    #sat_nums = ["f16"]
    #mlts=[18]
    hemi="north"
    imf_time_delay=15

    file_dir = "../../data/ssusi/" 

######################################

#    dfs = []
#    for i in range(len(sat_nums)):
#        sat_num = sat_nums[i]
#        mlt = mlts[i]
#
#        # Read the original data into a DF
#        df_ssusi = read_ssusi_aurora_data(sdate, edate, file_dir=file_dir,
#                                          sat_num=sat_num, hemi=hemi)
#
#        # Extract Auroral MLAT boundary at a given mlt
#        df_mlt = extract_aurora_boundary(df_ssusi, mlt=mlt)
#
#        # Add AU, AL into the dataframe
#        df_ae = add_au_al(df_mlt)
#
#        # add IMF
#        df = add_imf(df_ae, time_delay=imf_time_delay)
#
#        dfs.append(df)
#
#    df_all = pd.concat(dfs)
#
#    # Cleanup the data
#    dfn = prepare_data_for_project(df_all)
#
#    dfn.to_csv("~/Dropbox/Courses/STAT5214G/cleaned_data_more_param.csv")

##################################################
    # Plot the data
    sdate = dt.datetime(2011,1,1)
    edate = dt.datetime(2011,1,1)
    sat_nums= ["f16", "f17", "f18", "f19"]
    hemi = "north"

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

    for sat_num in sat_nums:
        # Read the original data into a DF
        df_ssusi = read_ssusi_aurora_data(sdate, edate, file_dir=file_dir,
                                          sat_num=sat_num, hemi=hemi)
        df_ssusi = df_ssusi.head(1)
        plot_auroral_boundary(ax, df_ssusi, hemi=hemi, linestyle="", marker="o",
                              markersize=0.5, mec="k", mfc="k", alpha=1.0)
    plt.show()

