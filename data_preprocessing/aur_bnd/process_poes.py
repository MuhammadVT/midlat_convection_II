"""
Finds fitted Aur Bnd circles and writes them into files.
"""

import datetime
import numpy as np
from poes import get_aur_bnd
#import sys
#sys.path.append("/home/muhammad/softwares/sataurlib/poes")
#from get_aur_bnd import PoesAur

sTimePOES = datetime.datetime( 2014, 3, 31 )
eTimePOES = datetime.datetime( 2018, 7, 1 )
dayCount = (eTimePOES - sTimePOES).days + 1

# Set the time interval for saving the output data
timeInterval=datetime.timedelta(minutes=10)
remove_outliers=True
cutoff_iqr_prop=1.5
save_fit_coeff = True
#fit_outdir = "../../data/poes/bnd_tmp/"
fit_outdir = "../../data/poes/bnd_fitcoeff/"

# Initialize the PoesAur Object attributes
# set up a few constants
minCutoffFitLat = 45.
delTimeCutOffNrstPass = 30 # min  # NOTE: The total time interval is twice of this value
mlonDiffOtrEndCutoff = 50.
delLatCutoff = 2.
# Set some parameters for gaussian fitting!
gauss_smooth_sigma = 1#2 
diffElctrCutoffBnd = 0.1#0.15
# More than an order of magnitude, remember its a log scale
filtEleFluxCutoffMagn = 1.25

def process_unit(inpDate):
    # Read data from the POES files
    # and get the auroral boundary location
    # by fitting a circle
    poesRdObj = get_aur_bnd.PoesAur()
    #poesRdObj = PoesAur()
    poesRdObj.minCutoffFitLat = minCutoffFitLat
    poesRdObj.delTimeCutOffNrstPass = delTimeCutOffNrstPass # min
    poesRdObj.mlonDiffOtrEndCutoff = mlonDiffOtrEndCutoff
    poesRdObj.delLatCutoff = delLatCutoff
    # Set some parameters for gaussian fitting!
    poesRdObj.gauss_smooth_sigma = gauss_smooth_sigma 
    poesRdObj.diffElctrCutoffBnd = diffElctrCutoffBnd
    # More than an order of magnitude, remember its a log scale
    poesRdObj.filtEleFluxCutoffMagn = filtEleFluxCutoffMagn

    ( poesAllEleDataDF, poesAllProDataDF ) = poesRdObj.read_poes_data_files(\
						poesRawDate=inpDate,\
						poesRawDir="../../data/poes/raw/" )
    # Or you can uncomment the line below and read the data!
    # ( poesAllEleDataDF, poesAllProDataDF ) = poesRdObj.read_poes_data_files(poesFiles)
    # Get for a given time get the closest satellite passes
    # We can do this at multiple instances for a given time range/step
    if not poesAllEleDataDF.empty:
        timeRange = [ poesAllEleDataDF["date"].min(),\
                         poesAllEleDataDF["date"].max() ]
        # Set the minutes of the starting and ending time to miltiples of 5
        t0, t1 = timeRange
        timeRange[0] = t0.replace(minute=5*int(np.floor(t0.minute/5.))) 
        timeRange[1] = t1.replace(minute=5*int(np.floor(t1.minute/5.))) 
        # aurPassDF contains closest passes for a given time 
        # for all the satellites in both the hemispheres!
        aurPassDF = poesRdObj.get_closest_sat_passes( poesAllEleDataDF,\
                                            poesAllProDataDF, timeRange, timeInterval=timeInterval)
        # determine auroral boundaries from all the POES satellites
        # at a given time. The procedure is described in the code! 
        # go over it!!!
        eqBndLocsDF = poesRdObj.get_nth_ele_eq_bnd_locs(aurPassDF, poesAllEleDataDF,
                                                        remove_outliers=remove_outliers,
                                                        cutoff_iqr_prop=cutoff_iqr_prop)
        # to get an estimate of the auroral boundary! fit a circle
        # to the boundaries determined from each satellite!
        # The fits are written to a file and can be stored in 
        # a given location
        # NOTE : set a proper outdir otherwise the data
        # is saved in the working directory by default

        try:
            bndDF=poesRdObj.fit_circle_aurbnd(eqBndLocsDF, save_to_file=True,
                                              save_fit_coeff=save_fit_coeff,
                                              fileFormat="txt",
                                              outDir=fit_outdir)
            print "ESTIMATED BOUNDARY"
            print bndDF.head()
            print "ESTIMATED BOUNDARY"
        except:
            print "Fitting Failed"

    return

def worker(dates):
    for inpDate in dates:
        process_unit(inpDate)
    return

def main(run_in_parallel=True):
    import multiprocessing as mp

    njobs = 10
    dts = [sTimePOES +  datetime.timedelta(n) for n in range(dayCount)]

    # Run in parallel
    if run_in_parallel:
        batch_size = len(dts)/njobs
        batch_idxs = range(0, len(dts), batch_size) 
        batch_idxs[-1] =  len(dts)
        dts_list = [dts[batch_idxs[i]:batch_idxs[i+1]] for i in range(len(batch_idxs)-1)]
        procs = []
        for dts_p in dts_list:
            p = mp.Process(target=worker, args=(dts_p,))
            procs.append(p)

            # run the process
            p.start()
        for p in procs:
            p.join()

    else:
        # Loop through the days and process files
        for inpDate in dts:
            # Run in sequence 
            process_unit(inpDate)


if __name__ == "__main__":
    #main(run_in_parallel=False)
    main(run_in_parallel=True)


