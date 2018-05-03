'''
Created on Jul. 18, 2017

Muhamamd
'''

import pdb

class los_data_to_db(object):
    """ A class that reads and holds sd LOS data from a given radar.
    It writes the data into a sqlite db using its move_to_db method."""

    def __init__(self, rad, stime, etime, ftype="fitacf", channel=None):

        """ 
        Parameters
        ----------
        rad : str
            Three-letter code for a rad
        stime : datetime.datetime
        etime : datetime.datetime
        ftype : str, default to "fitacf"
            SuperDARN file type. Valid inputs are "fitacf", "fitex"

        Returns
        -------
        los_file_to_db object 
            
        """

        import sys
        from read_sddata import read_data_from_file
        import datetime as dt
        
        # build the attributes
        self.rad = rad
        self.ftype = ftype
        self.stime = stime 
        self.etime = etime 
	self.channel = channel

        # read data from file 
        self.data = read_data_from_file(rad, stime, etime, ftype=self.ftype,
					channel=channel,
                                        tbands=None, coords="geo")

    def move_to_db(self, conn):
        """ writes the data into sqlite db

        Parameters
        ----------
        conn : sqlite3.connect

        Returns
        -------
        Nothing
        """

        import json 
        import sqlite3

        cur = conn.cursor()


	# create a table in sqlite db
	table_name = self.rad
	command = "CREATE TABLE IF NOT EXISTS {tb} (" +\
		  "vel TEXT, slist TEXT, gflg TEXT," +\
		  "bmnum INTEGER, bmazm REAL, nrang INTEGER, " +\
		  "rsep REAL, frang REAL, stid INTEGER, "+\
		  "datetime TIMESTAMP, "+\
		  "PRIMARY KEY(datetime, bmnum))".format(tb=table_name)
	cur.execute(command)

	# Write the data into table_name in the sqlite db

	data_dict = self.data
	for i in xrange(len(data_dict['datetime'])):
	    command = "INSERT OR IGNORE INTO {tb} (vel, slist, gflg, bmnum,"+\
		      "bmazm, nrang, rsep, frang, stid, datetime) "=\
	              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"\
		      .format(tb=table_name)
	    cur.execute(command, (json.dumps(data_dict["vel"][i]),
			json.dumps(data_dict["slist"][i]), json.dumps(data_dict["gflg"][i]),
			data_dict["bmnum"][i], data_dict["bmazm"][i], data_dict["nrang"][i],
			data_dict["rsep"][i],data_dict["frang"][i],data_dict["stid"][i],
			data_dict["datetime"][i]))

        # commit the change, once at one day of data points
        conn.commit()

def worker(dbName, baseLocation, rad, stime, etime, ftype, channel):
    """ A worker function used for multiprocessing.
    """

    import sqlite3
    import datetime as dt

    # make a db connection
    conn = sqlite3.connect(baseLocation + dbName)

    # collect the data 
    t1 = dt.datetime.now()
    print "creating an object for " + rad + " for " + \
	  str(stime) + "--" +  str(etime)
    obj = los_data_to_db(rad, stime, etime, ftype=ftype, channel=channel)
    print "object created for " + rad
    if obj.data is not None:
        # move data to db
        obj.move_to_db(conn)
        print ("object has been moved to db")

    t2 = dt.datetime.now()
    print ("creating and moving object to the db took " +\
	  str((t2-t1).total_seconds() / 60.)) + " mins\n"

    # close db connection
    conn.close()

def main():
    """ Call the functions above. Acts as an example code.
    Multiprocessing has been implemented to do parallel computing.
    The unit process is reading data from a given radar """
    
    import datetime as dt
    import sqlite3
    import multiprocessing as mp
    import sys
    import numpy as np

    # initialize parameters
    # NOTE: Do not forget to set the channel
    rads = ["wal", "bks", "fhe", "fhw", "cve", "cvw", "ade", "adw"]
    channel = [None, None, None, None, None, None, 'all', 'all']
    ftype = "fitacf"
    #ftype = "fitex"
    dbName =  "sd_los_vel.sqlite"

    stms = [dt.datetime(2014, 12, 16, 13, 30)]
    etms = [dt.datetime(2014, 12, 16, 14, 30)]

    # make a db connection
    baseLocation = "../data/sqlite3/"

    # loop through the datetimes in stms
    for i in range(len(stms)):
	stm = stms[i]
	etm = etms[i]
	
        # loop through the radars
        for rad in rads:
            # Store multiprocesses in a list
            procs = []
            for rad in rads:

	        worker(dbName, baseLocation, rad, stime, etime, ftype, channel)

#                # Creat a processe
#                p = mp.Process(target=worker,
#                               args=(dbName, baseLocation, rad, stm, etm,
#			       ftype, channel))
#                procs.append(p)
#
#                # Run the process
#                p.start()
#
#            # Make sure the processes terminate
#            for p in procs:
#                p.join()

    return


if __name__ == '__main__':
    main()

