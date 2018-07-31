# load modules
import sys
sys.path.append("../classification_of_HF_radar_backscatter")
from iscat_identifier import prepare_file
import datetime as dt
from davitpy.pydarn.sdio.fetchUtils import fetch_local_files
import os
import multiprocessing as mp
import logging
#davitpy.rcParams['verbosity'] = "debug"

def do_boxcarfiltering(sctr_day, ectr_day, rad_list, ftype, channel,
                       n_jobs=None, run_in_parallel=True):

    """ fetches, concatenates and does boxcar median filtering for data
    in the time iterval between sctr_day, ectr_day for a given radar.
    Multiprocessing is implemented in the code,
    where a day of data from a radar is defined as a process.

    Parameters
    ----------
    sctr_day : datetime.datetime
        Start day
    ectr_day : datetime.datetime
        End day
    rad_list : list
        Three-letter radar codes
    ftype : str
        Data file type. e.g., "fitacf", "fitex"
    channel : str
        radar channel. e.g., "a", "b", "c", "d", or "." which is all
    n_jobs : int or None
        Number of jobs that run in parallel.
        Default to None, in which case all the CPUs but one will used.

    Return
    ------
    Nothing
        
    """
    
    # calculate number of days 
    sctr_day = dt.datetime(sctr_day.year, sctr_day.month, sctr_day.day)
    ectr_day = dt.datetime(ectr_day.year, ectr_day.month, ectr_day.day)
    num_days = (ectr_day - sctr_day).days + 1  # includes the end day
    dt_range = [dt.timedelta(days=i)+sctr_day for i in xrange(num_days)]


    # Define an output queue
    output = mp.Queue()

    # number of jobs to be run in parallel
    if not n_jobs:
        # get the number of CUPs
        n_jobs = mp.cpu_count() - 1

    # loop throughs the radars
    for rad in rad_list:

        # create tmpdir to store the data
        rad_tmp_dir = "../data/" + rad + "_tmp/"
        if not os.path.exists(rad_tmp_dir):
            os.system("mkdir -p " + rad_tmp_dir)
        #rad_tmp_dir = os.getcwd() + "/data/"+ rad_tmp_dir + "/"
        print rad_tmp_dir + " is created"


        # cteate tmpdirs, one for each n_jobs
        tmp_dirs = []
        if run_in_parallel:
            n_dirs = n_jobs
        else:
            n_dirs = 1 
        for j in range(n_dirs):
            # create tmpdirs to store the data, one tmpdir for each process
            tmp_dir = "../data/" + rad + "_tmp" + "_" + str(j) +"/"
            os.system("mkdir -p " + tmp_dir)
            #tmp_dir = os.getcwd() + "/data/"+ tmp_dir + "/"
            print tmp_dir + " is created"
            tmp_dirs.append(tmp_dir)
        
        # iter through the days, one day at a time
        i = 0
        while 1:
            if run_in_parallel:
                # run n_jobs in parallel
                dts_tmp = dt_range[i*n_jobs:(i+1)*n_jobs]
                i = i + 1
                if len(dts_tmp) == 0:
                    break
                else:
                    procs = []
                    # send jobs in parallel
                    for j, ctr_dtm in enumerate(dts_tmp):
                       p = mp.Process(target=worker, args=(rad, ctr_dtm, ftype, channel, tmp_dirs[j]))
                       procs.append(p)
                       p.start()
            else:
                ctr_dtm = dt_range[i]
                i = i + 1
                worker(rad, ctr_dtm, ftype, channel, tmp_dirs[0])
                if i >= len(dt_range):
                    break

            if run_in_parallel:
                # exit the completed processes
                for p in procs:
                    p.join()

        # move processed data from tmpdirs of all processes to a single tmpdir
        for j in range(n_dirs):
            os.system("mv " + tmp_dirs[j] + "* " + rad_tmp_dir)

        # remove tmpdirs
        os.system("rm -rf " + "../data/" + rad + "_tmp_*")
        print "tmpdirs have been deleted"

    return

def worker(rad, ctr_dtm, ftype, channel, tmp_dir):
    """ A worker function fetches a one day worthy of data, 
    concatenate them and do boxcar median filtering.
    
    Parameters
    ----------
    rad : str
	Three-letter radar code
    ctr_dtm : datetime.datetime
    ftype : str
        Data file type. e.g., "fitacf", "fitex"
    channel : str
        radar channel. e.g., "a", "b", "c", "d", or "." which is all.
    tmp_dir : str
        temprory folder to store the processed data

    Return
    ------
    Nothing
            
    """

    print "start boxcarfiltering for data on ", ctr_dtm
    # concat 1 day of data, do boxcar filtering 
    scr = "local"
    localdict = {"ftype" : ftype, "radar" : rad, "channel" : channel}
    localdirfmt = "/sd-data/{year}/{ftype}/{radar}/"
    fnamefmt = ['{date}.{hour}......{radar}.{channel}.{ftype}',
                '{date}.{hour}......{radar}.{ftype}']

    ffname = prepare_file(ctr_dtm, localdirfmt, localdict, tmp_dir,
                          fnamefmt, oneday_file_only=True)
    print "end boxcarfiltering for data on ", ctr_dtm
    print "generated ", ffname

    # remove the original not filtered data
    if ffname is not None:
        os.system("rm " + ffname[:-1])
    else:
        pass

    return

def main(run_in_parallel=True):
    """Executes the codes above"""

    # input parameters
    # create center datetimes 
    """ NOTE: Set the channel value correctly. """
    
    n_jobs = 20
    #yrs = [2015, 2016]
    yrs = [2017]
    for yr in yrs:
        ftype = "fitacf"
        #ftype = "fitex"
        sctr_day = dt.datetime(yr,1,1)
        ectr_day = dt.datetime(yr+1,7,1)
        #ectr_day = dt.datetime(yr,7,1)

        #rad_list = ["bks", "wal", "fhe", "fhw", "cve", "cvw"]
        rad_list = ["wal", "fhe", "fhw", "cve", "cvw"]
        #rad_list = ["bks",  "wal", "fhe"]
        #rad_list = ["fhw", "cve", "cvw"]
        #rad_list = ["bks"]
        channel = None
        #channel = '.'

        do_boxcarfiltering(sctr_day, ectr_day, rad_list, ftype, channel,
                           n_jobs=n_jobs, run_in_parallel=run_in_parallel)

    return

if __name__ == "__main__":
    #main(run_in_parallel=False)
    main(run_in_parallel=True)


