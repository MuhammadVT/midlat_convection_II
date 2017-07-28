'''
Created on july. 28, 2017

Muhamamd
'''

import pdb

class iscat(object):

    def __init__(self, ctr_date, localdict, tmpdir=None,
               params=["velocity"], low_vel_iscat_event_only=True,
               search_allbeams=True, bmnum=7, no_gscat=True,
	       data_from_db=True, ffname=None):

        """ A class that holds the iscat events that have been obtained by
            data prepareting, reading data from a file or db,
            and searching for ionospheric scatter events.
        
        Parameters
        ----------
        ctr_date : datetime.datetime
            a full day for which iscat events are to be searched. 
        localdirc : dict
            holds radar related informations 
        tmpdir : str, default to None
            The directory used to store temporary files of boxcar filtered data. 
            If None, /tmp/sd will be used.
        params : list
            works for params=["velocity"] only
        low_vel_iscat_event_only : bool
            If set to True, returns low velocity ionospheric scatter events only
        search_allbeams : bool
            if set to true, iscat event searching will be performed on all the 
            beams, and ignores the bmnum argument. 
        bmnum : int
            bmnum argument only works in search_allbeams is set to False
        no_gscat : bool
            removes all the gscat (ground scatter)
	data_from_db : bool
	    If set to True data will be read from MySQL db.
        ffname : string
            The file full path for the boxcar filtered data.
            if ffname is not set to None, ffname will be constructed.

        Returns
        -------
        iscat object 
            A dict of dicts in the form of {bmnum:dict}.
            if no_gscat==False, returns data with all its points'
            gsflg marked as 1 (gscat) except for iscat.
            if no_gscat==True, returns only the iscat (gsflg=0)
            
        """

        import sys
        sys.path.append("../")
        from classification_of_HF_radar_backscatter.iscat_identifier import iscat_event_searcher
        import datetime as dt
        
        # creates class attributes
        self.rad = localdict["radar"]
        self.ftype = localdict["ftype"]
        self.channel = localdict["channel"]
        self.ctr_date = dt.datetime(ctr_date.year, ctr_date.month, ctr_date.day)
        self.low_vel_iscat_event_only = low_vel_iscat_event_only 
        self.search_allbeams = search_allbeams
	self.data_from_db = data_from_db
	self.tmpdir = tmpdir
        if self.search_allbeams:
            self.bmnum = None
        else:
            self.bmnum = bmnum 
        self.no_gscat = no_gscat
        if ffname is None:
            self.ffname = self._construct_filename()
        else:
            self.ffname = ffname

        # collect the iscat events 
        self.events = iscat_event_searcher(ctr_date, localdict, tmpdir=tmpdir
                           params=params, low_vel_iscat_event_only=low_vel_iscat_event_only,
                           search_allbeams=search_allbeams, bmnum=bmnum,
                           no_gscat=no_gscat, data_from_db=data_from_db, ffname=self.ffname)

        # If there is no iscat points in an event for a day for a specific beam then
	# the output is {bm:None}. We remove thes None type events. 
        if self.events is not None:
            self.events = self._remove_None()

    def _remove_None(self):
	"""Removes None type events"""

        for bn in self.events.keys():
            if self.events[bn] is None:
                self.events.pop(bn)
        if self.events == {}:
            self.events = None
        return self.events

    def _construct_filename(self, basedir="../data/"):
        """ constructs the file pull for boxcar filtered data of interest. """

        import datetime as dt

        # create stm (start time) and etm (end time) for a day
        stm = self.ctr_date
        etm = self.ctr_date + dt.timedelta(days=1)

	# contruct the file pull path of boxcar filtered data
        ffname = stm.strftime("%Y%m%d.%H%M%S") + "." + \
                 etm.strftime("%Y%m%d.%H%M%S") + "." + \
                 self.rad + "." + self.ftype + "f"
        ffname = basedir + self.rad + "/" + ffname

        return ffname

    def join_list_as_str(self, sep=","):
        """ joins list entry to a string with a certain seperator 
        This is useful to enter list entriy as a whole into a db

	Parameters
	----------
	sep : str
            The deliminator between strings
        
        Returns
	-------
	Nothing
            change each list entry as a joined strings seperated by sep

        """
        
        # join a list entriy as string seperated by sep
        for bn in self.events.keys():

            # find the variable names whose values are to be joined
            kys_tmp = self.events[bn].keys()
            kys = []

            # find out keys shows entries are lists
            for ky in kys_tmp:
                if isinstance(self.events[bn][ky][0], list):
                    kys.append(ky)

            # join the elements in a list into a string
            for ky in kys:
                self.events[bn][ky] = [sep.join([str(round(itm,2)) for itm in x])\
                        for x in self.events[bn][ky]]
        return
    
    def move_to_db(self, config_filename="../mysql_dbconfig_files/config.ini",
                   section="midlat", db_name=None):
        """ writes the data into a MySQL db given by conn argument.

        Parameters
        ----------
        config_filename: str
            name and path of the configuration file
        section: str, default to mysql
            section of database configuration
	db_name : str, default to None
            Name of the MySQL db to which iscat data will be written

        Returns
        -------
        Nothing
	
        """

        from mysql.connector import MySQLConnection

        # make db connection
        conn = MySQLConnection(dataname=db_name, **config_info)
	cur = conn.cursor()

        # loop through each radar beam
        for bmnum in self.events.keys():
            data_dict = self.events[bmnum]
            table_name = self.rad + "_bm" + str(bmnum)
            td = TD(table_name, column_map)

            # create a table
            table_name = self.rad + "_bm" + str(bmnum)
            command = "CREATE TABLE IF NOT EXISTS {tb} (\
                      vel TEXT DEFAULT NULL,\
                      rsep TINYINT(4) DEFAULT NULL,\
                      frang SMALLINT(4) DEFAULT NULL,\
                      bmazm FLOAT(7,2) DEFAULT NULL,\
                      slist TEXT DEFAULT NULL,\
                      datetime DATETIME,\
                      PRIMARY KEY (datetime))".format(tb=table_name)
            try:
                cur.execute(command)
            except Exception, e:
                logging.error(e)


            # loop through each scan time, usually 2 minutes,
            # and write the data into table_name in db
            for i in xrange(len(data_dict['datetime'])):
                command = "INSERT IGNORE INTO {tb} (vel, rsep, frang, bmazm, " +\
                          "slist, datetime) VALUES (%s, %s, %s, %s, %s, %s)"
                command = command.format(tb=table_name)
                try:
                    cur.execute(command, (data_dict["datetime"][i],
                                data_dict["vel"][i], data_dict["slist"][i],
                                data_dict["rsep"][i], data_dict["frang"][i],
                                data_dict["bmazm"][i]))
                except Exception, e:
                    logging.error(e)

        # commit the change, once at one-day of data points
        try:
            conn.commit()
        except Exception, e:
            logging.error(e)

        # close db connection
        conn.close()

def worker(rad):
    
    import datetime as dt
    import sys
    sys.path.append("../")

    # collect the iscat events 
    t1 = dt.datetime.now()
    print "creating an iscat object from " + rad + " for " + str(ctr_date)
    print "searching all beams of " + rad
    iscat_events = iscat(ctr_date, localdict,
                         params=params, low_vel_iscat_event_only=low_vel_iscat_event_only,
                         search_allbeams=search_allbeams, no_gscat=no_gscat, ffname=None)


    if iscat_events.events is not None:
        # join a list entriy as string seperated by sep
        iscat_events.join_list_as_str(sep=",")

        # move iscat events to db
        #t1 = dt.datetime.now()
        iscat_events.move_to_db(conn, column_map)
        #t2 = dt.datetime.now()
        #print ("move_to_db takes " + str((t2-t1).total_seconds() / 60.)) + " mins"
        print ("iscat has been moved to db")
    else:
        print "iscat_events.events is None"

    t2 = dt.datetime.now()
    print ("Finishing an iscat object took " + str((t2-t1).total_seconds() / 60.)) + " mins\n"


def main():

    import multiprocessing as mp

    rads_list = [["bks", "wal", "fhe"], ["fhw", "cve", "cvw"]]
    #rads_list = [["fhw", "cve", "cvw"]]
    

    # input parameters
    channel = None
    params=['velocity']
    ftype = "fitacf"
    #ftype = "fitex"
    #low_vel_iscat_event_only=False
    low_vel_iscat_event_only=True
    search_allbeams=True
    no_gscat=True

    # loop through time interval
    for dd in range(len(stms)):
        sdate = stms[dd]
        edate = etms[dd]
                
        num_days = (edate - sdate).days + 1
        dtm_range = [sdate + dt.timedelta(days=i) for i in xrange(num_days)]
       
        # loop through the radars
        for rad in rads:
            localdict = {"ftype" : ftype, "radar" : rad, "channel" : channel}

            # loop through dates:
            for ctr_date in dtm_range:

                worker()

    jobs = []
    for i in range(len(rads_list)):
        #worker(rads_list[i], season, baseLocation)
        p = mp.Process(target=worker, args=(rads_list[i], season, baseLocation))
        jobs.append(p)
        p.start()



    return

if __name__ == '__main__':
    main()
