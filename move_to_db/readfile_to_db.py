'''
Created on July. 11, 2017

Muhamamd
'''

import pdb

class read_file_to_db(object):
    """ A class that holds the boxcar median filtered data after reading them from a file.
    It writes the data into a MySQL db using its move_to_db method."""

    def __init__(self, rad, ctr_date, ftype="fitacf", params=["velocity"], ffname=None):

        """ 
        Parameters
        ----------
        rad : str
            Three-letter code for a rad
        ctr_date : datetime.datetime
            a full day for which data are to be read. 
        ftype : str, default to "fitacf"
            SuperDARN file type. Valid inputs are "fitacf", "fitex"
        params : list
            NOTE: works for params=["velocity"] only
        ffname : string, default to None
            Full path of a file to be read. if ffname is not set to None, 
            ffname will be constructed.

        Returns
        -------
        read_file_to_db object 
            A dict of dicts in the form of {bmnum:dict}, which holds a 
            one day worth of data from all beams of a certain radar.
            
        """

        import sys
        sys.path.append("../")
        from dopsearch_py.dopsearch import read_file
        import datetime as dt
        
        # build the attributes
        self.rad = rad
        self.ftype = ftype
        self.ctr_date = dt.datetime(ctr_date.year, ctr_date.month, ctr_date.day)
        if ffname is None:
            # construct ffname (file full path)
            self.ffname = self._construct_filename()
        else:
            self.ffname = ffname

        # create stm (start time) and etm (end time) 
        stm = self.ctr_date
        # add two minute to the etm to read the last record from
        # the boxcar filtered concatenated fitacf(ex) files
        etm = self.ctr_date + dt.timedelta(days=1) + dt.timedelta(minutes=2)

        # read data from file 
        # Note: data_from_db and plotrti arguments have to be False
        self.data = read_file(self.ffname, rad, stm, etm, params,
                              ftype=self.ftype, data_from_db=False,
                              plotrti=False)

    def _construct_filename(self, basedir="../data/"):
        """ constructs filename with full path for a file of interest

        Parameters
        ----------
        basedir : str
            Relative path for data files

        Returns
        -------
        str
            Filename with its full path
        
        """

        import datetime as dt

        # create stm (start time) and etm (end time) 
        stm = self.ctr_date
        etm = self.ctr_date + dt.timedelta(days=1)

        # consturc a file name with full path
        ffname = stm.strftime("%Y%m%d.%H%M%S") + "." + \
                 etm.strftime("%Y%m%d.%H%M%S") + "." + \
                 self.rad + "." + self.ftype + "f"
        ffname = basedir + self.rad + "/" + ffname

        return ffname

    def move_to_db(self, conn):
        """ writes the data into sqlite db

        Parameters
        ----------
        conn : MySQLdb.connect

        Returns
        -------
        Nothing
        """

        import json 
        import sqlite3

        cur = conn.cursor()

        # loop through all the beams
        for bmnum in self.data.keys():
            data_dict = self.data[bmnum]

            # create a table in sqlite db
            table_name = self.rad + "_bm" + str(bmnum)
            command = "CREATE TABLE IF NOT EXISTS {tb} (\
                      vel TEXT, rsep REAL, frang REAL, bmazm REAL,\
                      slist TEXT, gsflg TEXT,\
                      datetime TIMESTAMP PRIMARY KEY)".format(tb=table_name)
            cur.execute(command)

            # loop through each scan time, usually 2 minutes,
            # and write the data into table_name in the sqlite db
            for i in xrange(len(data_dict['datetime'])):
                command = "INSERT OR IGNORE INTO {tb} (vel, rsep, frang, bmazm,\
                            slist, gsflg, datetime) VALUES (?, ?, ?, ?, ?, ?, ?)"\
                            .format(tb=table_name)
                cur.execute(command, (json.dumps(data_dict["vel"][i]), data_dict["rsep"][i],
                            data_dict["frang"][i], data_dict["bmazm"][i],
                            json.dumps(data_dict["slist"][i]),
                            json.dumps(data_dict["gsflg"][i]),\
                            data_dict["datetime"][i]))

        # commit the change, once at one day of data points
        conn.commit()

def worker(dbName, baseLocation, rad, bmnum, ctr_date, 
           ftype, params, ffname):
    """ A worker function used for multiprocessing.
    """

    import mysql.connector import MySQLConnection, Error
    import datetime as dt
    # make a db connection
    conn = sqlite3.connect(baseLocation + dbName)

    # collect the data 
    t1 = dt.datetime.now()
    print "creating an object for " + rad + " for " + str(ctr_date)
    rf = read_file_to_db(rad, ctr_date, ftype=ftype, params=params, ffname=ffname)
    print "created an object for " + rad + " for " + str(ctr_date)
    if rf.data is not None:

        # move data to db
        rf.move_to_db(conn)
        print ("object has been moved to db")

    t2 = dt.datetime.now()
    print ("creating and moving object to the db took " +\
	    str((t2-t1).total_seconds() / 60.)) + " mins\n"

    # close db connection
    conn.close()

def main():
    """ Call the functions above. Acts as an example code.
    Multiprocessing has been implemented to do parallel computing.
    The unit process is for reading a day worth of data for a given radar"""
    
    import datetime as dt
    import sqlite3
    import multiprocessing as mp

    # input parameters

    # run the code for the following radars in parallel
    #rad_list = ["bks", "wal", "fhe", "fhw", "cve", "cvw"]
    channel = None
    params=['velocity']
    ftype = "fitacf"
    #ftype = "fitex"
    ffname = None

    # make a db connection
    baseLocation = "../data/sqlite3/"

            # make a db connection
            dbName = rad + "_" + ftype + ".sqlite"
            conn = sqlite3.connect(baseLocation + dbName)

            # close db connection
            conn.close()

    # loop through the dates in all_dates 
    for ctr_date in all_dates:

        # loop through the radars
        for rad in rad_list:

            # Store multiprocesses in a list
            procs = []
            for rad in [rad]:

                # Skip the date if rad has not data for that date
                if ctr_date not in rads_dict[rad]['date']:
                    continue


                # Creat a processe
                idx_tmp = rads_dict[rad]['date'].index(ctr_date)
                bmnum = rads_dict[rad]['beam'][idx_tmp]
                dbName = rad + "_" + ftype + ".sqlite"
                p = mp.Process(target=worker,
                               args=(dbName, baseLocation, rad, bmnum, ctr_date,
                                     ftype, params, ffname))
                procs.append(p)

                # Run the process
                p.start()

            # Make sure the processes terminate
            for p in procs:
                p.join()

    return

if __name__ == '__main__':
    main()

