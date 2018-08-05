#import matplotlib
#matplotlib.use("Agg")


def read_ssusi_aurora_data(stime, etime, orbit, file_dir="../../data/ssusi/",
                           file_name=None):
    """Reads DMSP SSUSI EDR-AUR data type"""
    import netCDF4
    import datetime as dt
    import numpy as np
    import pandas as pd

    if file_name is None:
        file_dir = file_dir + stime.strftime("%Y%m%d") + "/"
        file_name = file_dir + "ssusi_aurora_" + stime.strftime("%Y") +\
                    "_" + stime.strftime("%j") + ".ncdf"

    # Convert data format from netCDF4 to pandas DataFrame
    ds = netCDF4.Dataset(file_name)
    orbit_indx = np.where(ds.variables["Orbit Number"][:] == orbit)[0][0]
    magnetic_north_latitude = ds.variables["Magnetic North latitude"][orbit_indx,:]
    magnetic_north_local_time = ds.variables["Magnetic North Local Time"][orbit_indx,:]
    magnetic_north_mean_energy = ds.variables["Magnetic North Mean Energy"][orbit_indx,:]   # Kev
    magnetic_north_flux = ds.variables["Magnetic North Flux"][orbit_indx,:]     # ergs/s/cm2
    ut_seconds =  ds.variables["Magnetic North UT second"][orbit_indx,:]
    sdate = dt.datetime(stime.year, stime.month, stime.day)
    dtms = [sdate + dt.timedelta(seconds=x) for x in ut_seconds.astype("float")]

    # Calculate number density
    magnetic_north_number_flux = (magnetic_north_flux * 6.2415e11) / \
                                 (magnetic_north_mean_energy * 1.e3)             # #/s/cm2

    # Construct a dataframe
    df = pd.DataFrame(data={"datetime":dtms,
                            "mlat_north":magnetic_north_latitude,
                            "mlt_north":magnetic_north_local_time,
                            "JEe_north":magnetic_north_flux,
                            "AvgEe_north":magnetic_north_mean_energy,
                            "JNEe_north":magnetic_north_number_flux})
    
    # Select data for the time interval of interest
    df = df.loc[(df.datetime >= stime) & (df.datetime <= etime), :]

    return df

if __name__ == "__main__":

    import datetime as dt
    stime = dt.datetime(2013,3,17,16,24)
    etime = dt.datetime(2013,3,17,16,40)
    sat = "F16"
    #df = read_ssusi_aurora_data(stime, etime, orbit, file_name=None)


##########################################################
    import netCDF4


    file_dir = "../../data/ssusi/f" + sat[1:] +  "/" + stime.strftime("%Y%m%d") + "/"
    file_name = "PS.APL_V0105S024CE0018_SC.U_DI.A_GP.F16-SSUSI_PA.APL-EDR-AURORA_DD.20130317_SN.48568-00_DF.NC"

    # Convert data format from netCDF4 to pandas DataFrame
    ds = netCDF4.Dataset(file_dir+file_name)


