import datetime
from poes import dwnld_poes

# Download the raw poes files
sTimePOES = datetime.datetime( 2017, 1, 1 )
eTimePOES = datetime.datetime( 2018, 7, 1 )

#sTimePOES = datetime.datetime( 2015, 4, 9 )
#eTimePOES = datetime.datetime( 2015, 4, 9 )

# director to store raw poes files
dayCount = (eTimePOES - sTimePOES).days + 1

# Loop through the days and download files
for inpDate in (sTimePOES + \
		datetime.timedelta(n) for n in range(dayCount)):
    poesDwnldObj = dwnld_poes.PoesDwnld(inpDate)
    # NOTE : set a proper outdir otherwise the data
    # is saved in the working directory by default
    poesFiles = poesDwnldObj.get_all_sat_data(outDir="../../data/poes/raw_tmp")

