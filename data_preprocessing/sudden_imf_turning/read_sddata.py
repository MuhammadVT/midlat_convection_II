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
                try:
                    data['vel'].append([round(x, 2) for x in myBeam.fit.v])
                except TypeError:
                    myBeam = myPtr.readRec()
                    continue
                data['datetime'].append(myBeam.time)
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

if __name__ == "__main__":

    import datetime as dt
    stm = dt.datetime(2014, 12, 16, 13, 30)
    etm = dt.datetime(2014, 12, 16, 14, 30)
    #rad = "ade"
    #channel = "all"
    rad = "fhw"
    channel = None
    data_dict = read_data_from_file(rad, stm, etm, ftype="fitacf", channel=channel,
                                    tbands=None, coords="geo")
