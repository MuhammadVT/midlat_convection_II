The modules in this folder (midlat_convection_II folder) should be run in the following order:
   
    * boxcarfilter in data_processing folder to prepare the fitacf data by concatenating and
        boxcar filtering for a one day worthy of data at a time.

    * data_processing.readfile_to_db 
        reads the boxcar filtered data into db. No duplicate records can happen in this process
        because the datetime is set as PRIMARY KEY

    * data_processing.iscat_to_db
        implements the iscat search algorithm and stores the data into different db 
    
    * data_processing.calc_geolatc_geolonc    calculates the latc, lonc of the range-beam cells 
    * data_processing.geo_to_mlt      converts latc, lonc and bmazm of range-beam cells from "geo" to "mlt"
    * bin_data                   bins the data into latc-lonc-az bins
    NOTE: The processing is for a beam of certain radar at a time.
          The results are still in the scat db files.

    * data_processing.ten_min_median
        bins each ten-minute of data from a radar by choosing 
        the median vector in each azimuth bin within each grid cell. 
        The results are stored in different db files: one db table for a radar

    * data_processing.combine_ten_min_median
        combines the tables in ten_min_median db into a single table.

    * data_processing.gmi_imf.gmi_based_filter
        filteres the gridded iscat data based on geomagnetic indices
        NOTE: The following should be done before running gmi_based_filter
              run gmi_imf_to_db.py if indices data are not stored in db

    * data.binning.build_master_db
        combines all the ten-min median filtered radars' data into one master table.
        Datetime information is lost in the summary table. 
        The results are stored in a different db file named master_ftype.
        NOTE: this has two main functions: main() and main_imf().
        main() can be run here, but main_imf() should come after imf_based_filter below is
        completed.

    * data_processing.gmi_imf.imf_based_filter
        NOTE: The following should be done before running imf_based_filter
              run gmi_imf_to_db.py if imf data are not stored in db
              run gmi_imf_binning.py if imf is not yet binned

    * data.binning.build_master_db.main_imf()

    * data.binning.cos_fit
        Do the cosine fitting to the LOS data in master db.
        cosfit is performed for LOS velocities in each latc-lonc grid cell

