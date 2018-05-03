def read_data_from_file(rad, stm, etm, ftype="fitacf", channel=None,
                        tbands=None, coords="geo"):

    """Reads data from file for a given radar
    ----------
    rad : str
        Three-letter code for a rad
    ftype : str, default to "fitacf"
        SuperDARN file type. Valid inputs are "fitacf", "fitex"

    tbands : list
        a list of the frequency bands to separate data into
    coords : string 
        converts the range-time cell position (clat, clon) into the value 
        given by coords. Has to be one of ["mag", "geo", "mlt"]
        (Note: only works for "geo" so far due to speed issue)

    Returns
    -------
    A dictionary 

    Written by Muhammad 20180502

    """

    from davitpy import pydarn
    import copy
    from davitpy.pydarn.sdio import radDataOpen

    # read from a file
    myPtr = radDataOpen(stm, rad, eTime=etm, fileType=ftype, channel=channel)

    if tbands is None:
        tbands = [8000, 20000]

    # Initialization.
    data = dict() 

    # Parameters to read from dmap file
    data_keys = ['datetime', 'slist', 'vel', 'gflg', 'bmnum', 'bmazm',
                 'nrang', 'rsep', 'frang', 'stid']

    for d in data_keys:
        data[d] = []

    # return None if no data available
    try:
        myPtr.rewind()
    except:
        data = None
        return data

    # Read the parameters of interest.
    myBeam = myPtr.readRec()
    while(myBeam is not None):
        if(myBeam.time > myPtr.eTime): break
        if(myPtr.sTime <= myBeam.time):
            if (myBeam.prm.tfreq >= tbands[0] and myBeam.prm.tfreq <= tbands[1]):
                data['datetime'].append(myBeam.time)
                data['vel'].append(myBeam.fit.v)
                data['slist'].append(myBeam.fit.slist)
                data['gflg'].append(myBeam.fit.gflg)
                data['bmnum'].append(myBeam.bmnum)
                data['bmazm'].append(round(myBeam.prm.bmazm,2))
                data['nrang'].append(myBeam.prm.nrang)
                data['rsep'].append(myBeam.prm.rsep)
                data['frang'].append(myBeam.prm.frang)
                data['stid'].append(myBeam.stid)

        # Read data from next record
        myBeam = myPtr.readRec()

    # Set data to None if it is empty
    if data['datetime'] == []:
        data = None

    return data

def read_from_db(rad, bmnum, stm, etm, ftype="fitacf",
                 baseLocation="../data/sqlite3/"):

        """ reads the data from db instead of files
        NOTE : you need to bugfix this function        
        """

        import sqlite3
        import json
        import sys 
        sys.path.append("../")
        from move_to_db.month_to_season import get_season_by_month
        import datetime as dt

        # make a db connection
        dbName = rad + "_" + ftype + ".sqlite"
        season = get_season_by_month((stm+dt.timedelta(days=1)).month)
        baseLocation = baseLocation + season + "/original_data/"
        conn = sqlite3.connect(baseLocation + dbName, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()

        # get all the table names
        cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        tbl_names = cur.fetchall()
        tbl_names = [x[0] for x in tbl_names]

        # get the available beam numbers 
        beam_nums = [x.split("_")[-1][2:] for x in tbl_names]
        beam_nums = [int(x) for x in beam_nums]
    
        # loop through each table
        beams_dict = {}
        for jj, bmnum in enumerate(beam_nums):
            # get the data from db
            command = "SELECT * FROM {tb}\
                       WHERE datetime BETWEEN '{stm}' AND '{etm}'\
                       ORDER BY datetime".\
                       format(tb=tbl_names[jj], stm=stm, etm=etm)
            cur.execute(command)
            rws = cur.fetchall()
            if rws:
                data_dict = {}
                data_dict['vel'] = [json.loads(x[0]) for x in rws]
                data_dict['rsep'] = [x[1] for x in rws]
                data_dict['frang'] = [x[2] for x in rws]
                data_dict['bmazm'] = [x[3] for x in rws]
                data_dict['slist'] = [json.loads(x[4]) for x in rws]
                data_dict['gsflg'] = [json.loads(x[5]) for x in rws]
                data_dict['datetime'] = [x[6] for x in rws]
                beams_dict[bmnum] = data_dict
        if not beams_dict:
            beams_dict = None

        return beams_dict

if __name__ == "__main__":

    import datetime as dt
    stm = dt.datetime(2014, 12, 16, 13, 30)
    etm = dt.datetime(2014, 12, 16, 13, 50)
    rad = "cve"
    data_dict = read_data_from_file(rad, stm, etm, ftype="fitacf", channel=None,
                                    tbands=None, coords="geo")
