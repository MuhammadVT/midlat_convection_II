class beam_data():
    """a class to contain the data from a radar beam sounding,
  
    Attributes
    -----------
    stid : (int)
        radar station id number
    time : (datetime)
        timestamp of beam sounding
    bmnum : (int)
        beam number
    prm : parameters 
    fit : fitted parameters

    Written by muhammad 20180503
    """
    def __init__(self, beam_dict):
        #initialize the attr values
        self.time = beam_dict['datetime']
        self.bmnum = beam_dict['bmnum']
        self.stid = beam_dict['stid']
        self.fit = fit_data(beam_dict)
        self.prm = prm_data(beam_dict)

class fit_data():
    """a class to contain fited parameters from a radar beam sounding,

    Written by muhammad 20180503
    """
    def __init__(self, beam_dict):

        self.vel = beam_dict['slist']
        self.slist = beam_dict['slist']
        self.gflg = beam_dict['gflg']

class prm_data():
    """a class to contain parameters from a radar beam sounding,

    Written by muhammad 20180503
    """
    def __init__(self, beam_dict):

        self.bmazm = beam_dict['bmazm']
        self.nrang = beam_dict['nrang']
        self.rsep = beam_dict['rsep']
        self.frang = beam_dict['frang']

def read_beamdata_from_db(rad, stm, etm, dbName, ftype="fitacf",
                          baseLocation="../../data/sqlite3/"):

        """ Reads the data from db instead of files

        Returns
        -------
        A list of beam_data objects
        """

        import sqlite3
        import json
        import sys
        import datetime as dt

        # make a db connection
        conn = sqlite3.connect(baseLocation + dbName, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()

        table_name = rad
        beams_dict = {}
        # get the data from db
        command = "SELECT vel, slist, gflg, bmnum, "+\
                  "bmazm, nrang, rsep, frang, stid, datetime FROM {tb} "+\
                  "WHERE datetime BETWEEN '{stm}' AND '{etm}' "+\
                  "ORDER BY datetime"
        command = command.format(tb=table_name, stm=stm, etm=etm)
        cur.execute(command)
        rws = cur.fetchall()
        if rws:
            beams_dict['vel'] = [json.loads(x[0]) for x in rws]
            beams_dict['slist'] = [json.loads(x[1]) for x in rws]
            beams_dict['gflg'] = [json.loads(x[2]) for x in rws]
            beams_dict['bmnum'] = [x[3] for x in rws]
            beams_dict['bmazm'] = [x[4] for x in rws]
            beams_dict['nrang'] = [x[5] for x in rws]
            beams_dict['rsep'] = [x[6] for x in rws]
            beams_dict['frang'] = [x[7] for x in rws]
            beams_dict['stid'] = [x[8] for x in rws]
            beams_dict['datetime'] = [x[9] for x in rws]

        if not beams_dict:
            beams_data = None 
        else:
            # Construct beam_data objects
            #beams_data = [{beams_dict.keys[i]:beams_dict.values[i][j]\
            #              for i in range(len(beams_dict.keys()))} \
            #              for j in range(len(beams_dict['datetime']))]
            beams_data = []
            keys = beams_dict.keys()
            for k in range(len(beams_dict['datetime'])):
                values = [beams_dict[key][k] for key in keys]
                beam_data =  {keys[i]:values[i] for i in range(len(keys))}
                beams_data.append(beam_data)
            
        return beams_data

if __name__ == "__main__":

    import datetime as dt
    stm = dt.datetime(2014, 12, 16, 13, 30)
    etm = dt.datetime(2014, 12, 16, 14, 30)
    ftype = "fitacf"
    #rad = "ade"
    rad = "fhw"
    baseLocation="../../data/sqlite3/"
    dbName = "sd_los_data_" + ftype + ".sqlite"

    beams_data = read_beamdata_from_db(rad, stm, etm, dbName, ftype=ftype,
                                       baseLocation=baseLocation)

