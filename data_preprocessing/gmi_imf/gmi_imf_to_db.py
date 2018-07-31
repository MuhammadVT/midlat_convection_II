import pdb

class gmi_imf_to_db(object):

    def __init__(self, stm, etm, db_name=None,
                 base_location="../../data/sqlite3/"):
        """ Fetches geomagnetic indices and writes them into an sqlite db.

        Parameters
        ----------
        stm : datetime.datetime
            The start date
        etm : datetime.datetime
            The end date
        db_name : str, default to None
            Name of an sqlite db to which data will be written
        base_location : str
            Path to db_name file
        
        Returns
        -------
        Nothing
        
        """

        self.stm = stm
        self.etm = etm
        self.base_location = base_location

        # construct a db_name
        if db_name is None:
            self.db_name = "gmi_imf.sqlite"
        else:
            self.db_name = db_name

    def _create_dbconn(self):
        """make a db connection."""
        import sqlite3

        # make a db connection
        conn = sqlite3.connect(self.base_location + self.db_name,
                               detect_types = sqlite3.PARSE_DECLTYPES)
        return conn

    def imf_to_db(self, resolution):
        """fetches imf data and writes them to db """

        import datetime as dt
        from davitpy.gme.ind import readOmni
        from davitpy.gme.ind import readOmniFtp
        import numpy as np

        # read the data we want in GSM coords
        data_dict = {'datetime':[], 'Bx':[], 'By':[], 'Bz':[]}
        omni_list = readOmniFtp(sTime=self.stm, eTime=self.etm, res=resolution)
        data_dict['datetime'] = [omni_list[i].time for i in range(len(omni_list))]
        data_dict['Bx'] = [omni_list[i].bx for i in range(len(omni_list))]
        data_dict['By'] = [omni_list[i].bym for i in range(len(omni_list))]
        data_dict['Bz'] = [omni_list[i].bzm for i in range(len(omni_list))]

        # clock angle
        #data_dict['theta'] = np.degrees(np.arctan2(data_dict['By'],
        #                                           data_dict['Bz'])) % 360
        data_dict['theta'] = [round(np.degrees(np.arctan2(data_dict['By'][i],
                              data_dict['Bz'][i])) % 360, 2) \
                              if (data_dict['By'][i] is not None and \
                              data_dict['Bz'][i] is not None) \
                              else None for i in range(len(data_dict['By']))]
        
        # make db connection
        self.conn = self._create_dbconn()
        # create table
        table_name = "IMF"
        colname_type = "datetime TIMESTAMP PRIMARY KEY, Bx REAL, " +\
                       "By REAL, Bz REAL, theta REAL"
        command = "CREATE TABLE IF NOT EXISTS {tb} ({colname_type})"
        command = command.format(tb=table_name, colname_type=colname_type)
        self.conn.cursor().execute(command)
        
        # populate the table
        row_num = len(data_dict['datetime'])
        columns = "datetime, Bx, By, Bz, theta"
        for i in xrange(row_num):
            dtm = data_dict['datetime'][i]
            Bx = data_dict['Bx'][i]
            By = data_dict['By'][i]
            Bz = data_dict['Bz'][i]
            theta = data_dict['theta'][i]
            if (Bx is not None) and (By is not None) and (Bz is not None):
                command = "INSERT OR IGNORE INTO {tb}({columns}) " +\
                          "VALUES (?, ?, ?, ?, ?)"
                command = command.format(tb=table_name, columns=columns)
                self.conn.cursor().execute(command, (dtm, Bx, By, Bz, theta))
        self.conn.commit()

        # close db connection
        self.conn.close()

    def f107_to_db(self):
        """fetches F10.7 data and writes them to db """

        import datetime as dt
        import pandas as pd

        # read the data from txt file 
        date_parser = lambda x: pd.datetime.strptime(x, '%Y %j %H')
        fname = "../data/sqlite3/gmi_imf/F107.txt"
        df = pd.read_csv(fname, index_col=0, header=None, 
                         names=["Year", "DOY", "Hour", "F107"],
                         skipinitialspace=True, delim_whitespace=True,
                         parse_dates={'datetime': [0,1,2]},
                         date_parser=date_parser, na_values=999.9)

        data_dict = {}
        data_dict['datetime'] = df.index.to_pydatetime().tolist() 
        data_dict['F107'] = df.F107.tolist() 
        
        # make db connection
        self.conn = self._create_dbconn()
        # create table
        table_name = "F107"
        colname_type = "datetime TIMESTAMP PRIMARY KEY, F107 REAL"
        command = "CREATE TABLE IF NOT EXISTS {tb} ({colname_type})"
        command = command.format(tb=table_name, colname_type=colname_type)
        self.conn.cursor().execute(command)
        
        # populate the table
        row_num = len(data_dict['datetime'])
        columns = "datetime, F107"
        for i in xrange(row_num):
            dtm = data_dict['datetime'][i]
            F107 = round(data_dict['F107'][i],1)
            command = "INSERT OR IGNORE INTO {tb}({columns}) VALUES (?, ?)".\
                      format(tb=table_name, columns=columns)
            self.conn.cursor().execute(command, (dtm, F107))
        self.conn.commit()

        # close db connection
        self.conn.close()


    def kp_to_db(self, kp_lim=None):
        """fetches Kp data from GFZ Potsdam FTP server and writes them to db.
        Parameters
        ----------
        kp_lim : list
            The lower (>=) and upper (<) limit set for kp selection
        """

        from davitpy import gme
        import datetime as dt

        # make a db connection
        self.conn = self._create_dbconn()

        # fetch Kp data
        data_dict = {'datetime':[], 'kp':[]}

        # readKpFtp can not read across year boundaries. So we need to 
        # use a for-loop to avoid it.
        sdtm = self.stm
        while sdtm <= self.etm:
            if sdtm.year == self.etm.year:
                edtm = self.etm
            else:
                edtm = dt.datetime(sdtm.year, 12, 31) 
            Kp_list = gme.ind.readKpFtp(sTime=sdtm,eTime=edtm)

            # loop through each day withing sdtm and edtm
            day_num = (edtm-sdtm).days + 1
            for n in xrange(day_num):
                try:
                    kp_tmp = Kp_list[n].kp
                    time_tmp = Kp_list[n].time
                except:
                    print("Latest available Kp record is " + str(Kp_list[-1]))
                    break
                if kp_tmp is not None:
                    if len(kp_tmp) < 8 or len(kp_tmp) > 8:
                        print(str(len(kp_tmp))  + " values for the day " +\
                                  str(time_tmp) + ", should have 8 values")
                        continue
                    else:
                        for l in range(len(kp_tmp)):
                            if len(kp_tmp[l])== 2:
                                if kp_tmp[l][1] == '+':
                                    data_dict['kp'].append(int(kp_tmp[l][0])+0.3)
                                elif kp_tmp[l][1] == '-':
                                    data_dict['kp'].append(int(kp_tmp[l][0])-0.3)
                            else:
                                data_dict['kp'].append(int(kp_tmp[l][0]))
                            data_dict['datetime'].append(time_tmp + dt.timedelta(hours=3*l))

            # update sdtm to the first day of next year
            sdtm = edtm + dt.timedelta(days=1)

        # move to db
        if (data_dict['datetime'])!=[]:
            # create table
            table_name = "kp"
            colname_type = "datetime TIMESTAMP PRIMARY KEY, kp REAL"
            command = "CREATE TABLE IF NOT EXISTS {tb} ({colname_type})"
            command = command.format(tb=table_name, colname_type=colname_type)
            self.conn.cursor().execute(command)
            
            # populate the table
            row_num = len(data_dict['datetime'])
            columns = "datetime, kp"
            for i in xrange(row_num):
                dtm = data_dict['datetime'][i]
                k = data_dict['kp'][i]
                if kp_lim is not None:
                    if k < kp_lim[0] or k >= kp_lim[1]:
                        store_kp = False
                    else:
                        store_kp = True
                else:
                    store_kp = True
                if store_kp:
                    command = "INSERT OR IGNORE INTO {tb}({columns}) VALUES (?, ?)".\
                              format(tb=table_name, columns=columns)
                    self.conn.cursor().execute(command, (dtm, k))
            
            # commit the change
            self.conn.commit()

        # close db connection
        self.conn.close()
        
        return


    def symh_to_db(self, symh_lim=None):
        """fetches SYMH data and writes them to db """

        from davitpy import gme
        import datetime as dt

        # make a db connection
        self.conn = self._create_dbconn()

        # read SYMH data
        data_dict = {'datetime':[], 'symh':[]}
        sym_list = gme.ind.symasy.readSymAsy(sTime=self.stm,eTime=self.etm)
        for i in xrange(len(sym_list)):
            data_dict['symh'].append(sym_list[i].symh)
            data_dict['datetime'].append(sym_list[i].time)

        # move to db
        if (data_dict['datetime'])!=[]:

            # create table
            table_name = "symh"
            colname_type = "datetime TIMESTAMP PRIMARY KEY, symh REAL"
            command = "CREATE TABLE IF NOT EXISTS {tb} ({colname_type})"
            command = command.format(tb=table_name, colname_type=colname_type)
            self.conn.cursor().execute(command)
            
            # populate the table
            row_num = len(data_dict['datetime'])
            columns = "datetime, symh"
            for i in xrange(row_num):
                dtm = data_dict['datetime'][i]
                k = data_dict['symh'][i]
                if symh_lim is not None:
                    if k < symh_lim[0] or k >= symh_lim[1]:
                        store_symh = False
                    else:
                        store_symh = True
                else:
                    store_symh = True
                if store_symh:
                    command = "INSERT OR IGNORE INTO {tb}({columns}) VALUES (?, ?)".\
                              format(tb=table_name, columns=columns)
                    self.conn.cursor().execute(command, (dtm, k))

                    
                #col_value_map = {"datetime":data_dict['datetime'][i],
                #                 "vel":data_dict['vel'][i],
                #                 "slist":data_dict["slist"][i],
                #                 "bmazm":data_dict["bmazm"][i]} 
            self.conn.commit()

        # close db connection
        self.conn.close()

    def aualae_to_db(self, resolution=1):
        """fetches AE data and writes them to db """

        import ae
        import datetime as dt

        # make a db connection
        self.conn = self._create_dbconn()

	# create table
	table_name = "aualae"
	colname_type = "datetime TIMESTAMP PRIMARY KEY, au REAL, al REAL, ae REAL"
	command = "CREATE TABLE IF NOT EXISTS {tb} ({colname_type})"
	command = command.format(tb=table_name, colname_type=colname_type)
	self.conn.cursor().execute(command)

        # read AE data
        data_dict = {'datetime':[], 'au':[], 'al':[], 'ae':[]}
	# Need to loop through each year because of the max # of years <=1 limitation
	for yr in range(self.stm.year, self.etm.year+1):
	    stm_tmp = dt.datetime(yr, 1, 1)
	    etm_tmp = dt.datetime(yr, 12, 31)
	    # reads AE data
	    ae_list = ae.readAeWeb(sTime=stm_tmp, eTime=etm_tmp, res=resolution)
	    for i in xrange(len(ae_list)):
		data_dict['au'].append(ae_list[i].au)
		data_dict['al'].append(ae_list[i].al)
		data_dict['ae'].append(ae_list[i].ae)
		data_dict['datetime'].append(ae_list[i].time)

	    # move to db
	    if (data_dict['datetime'])!=[]:
		# populate the table
		row_num = len(data_dict['datetime'])
		columns = "datetime, au, al, ae"
		for i in xrange(row_num):
		    dtm = data_dict['datetime'][i]
		    au_i = data_dict['au'][i]
		    al_i = data_dict['al'][i]
		    ae_i = data_dict['ae'][i]
		    command = "INSERT OR IGNORE INTO {tb}({columns}) VALUES (?, ?, ?, ?)".\
			      format(tb=table_name, columns=columns)
		    self.conn.cursor().execute(command, (dtm, au_i, al_i, ae_i))

		self.conn.commit()
	    print("Stored data for Year " + str(yr))

        # close db connection
        self.conn.close()


def main():
    import datetime as dt
    #stm = dt.datetime(2010, 1, 1)
    stm = dt.datetime(2017, 1, 1)
    etm = dt.datetime(2018, 7, 2)
    db_name = None
    #db_name = "test_gmi_imf.sqlite"
    base_location = "../../data/sqlite3/"
    #base_location = "/home/sd-spare/muhammad/github_repos/convection_response_to_sudden_imf_turning/data/sqlite3/"

    kp_lim = None
    symh_lim = None
    dst_lim = None
    resolution = 1

    # create an object
    gmi = gmi_imf_to_db(stm, etm, db_name=db_name, base_location=base_location)

    # store IMF into db
    print "storing IMF to db"
    gmi.imf_to_db(resolution=resolution)
    print "imf is done"

#    # store F107 into db
#    print "storing F107 to db"
#    gmi.f107_to_db()
#    print "F107 is done"


#    # store kp into db
#    print "storing kp to db"
#    gmi.kp_to_db(kp_lim=kp_lim)
#    print "kp is done"

#    # store symh into db
#    print "storing symh to db"
#    gmi.symh_to_db(symh_lim=symh_lim)
#    print "symh is done"

#    # store AU, AL, AE into db
#    print "storing AU, AL, AE to db"
#    gmi.aualae_to_db(resolution=resolution)
#    print "AU, AL, AE is done"

if __name__ == "__main__":
    main()
