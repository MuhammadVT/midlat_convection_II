import os
import datetime
from imagers.ssusi import dwnld_ssusi
from imagers.ssusi import read_ssusi

# Download SSUSI data
currDate = datetime.datetime( 2011, 1, 30 )
endDate = datetime.datetime( 2012, 1, 1 )
tDelta = datetime.timedelta(days=1)
dataTypeList = [ "edr-aur" ]
#dataTypeList = [ "sdr", "edr-aur" ]
#dataTypeList = [ "edr-aur" ]
satList = ["f16", "f17", "f18", "f19"]
tempFileDir = "../../data/ssusi_tmp"# Make sure you have this dir or create it

ssDwnldObj = dwnld_ssusi.SSUSIDownload(outBaseDir = tempFileDir)
while currDate <= endDate:
    print "currently downloading files for --> ",\
        currDate.strftime("%Y-%m-%d")
    ssDwnldObj.download_files(currDate, dataTypeList, satList=satList)
    currDate = currDate + tDelta

    # scp data to sd-data
    os.system("scp -r " + tempFileDir + "/*" + "sd-data@sd-data1:/sd-data/dmsp/ssusi/")
    print ("files for "  + currDate.strftime("%Y-%m-%d") + " are moved to sd-data")
    os.system("rsync -avr " + tempFileDir + "/" + " --remove-source-files " +\
              "sd-data@sd-data1:/sd-data/dmsp/ssusi/")
    #os.system("rm -r " + tempFileDir + "/*")
    print ("files for "  + currDate.strftime("%Y-%m-%d") + " are removed from local directory")




