import os
import datetime
from imagers.ssusi import read_ssusi

# The directory where files are stored
rawFileDir = "../../data/ssusi/" # Make sure you have this dir or create it
prcsdFileDir = "../../data/ssusi/prcsd/"
satList = [ "f16", "f17", "f18", "f19" ]

currDate = datetime.datetime( 2013, 3, 17)
endDate = datetime.datetime(  2013, 3, 17)

data_type = "SDR"

tDelta = datetime.timedelta(days=1)
while currDate <= endDate:
    for currSat in satList:
        currDir = rawFileDir + currSat + "/"
        for root, dirs, files in os.walk(currDir):
            for nd, dd in enumerate(dirs):
                if currDate.strftime("%Y%m%d") not in dd:
                    continue
                print "processing data --> ",\
                         currDate.strftime("%Y-%m-%d"), " sat-->", currSat
                ssRdObj = read_ssusi.ProcessData( [root + dd + "/"],\
                             prcsdFileDir, currDate )
                ssRdObj.processed_data_to_file(keepRawFiles=True)
    currDate += tDelta


