"""
Created on July 20, 2017.
Muhammad
"""

class db_config():
    """
    Creats an object to handle config files for MySQL connection.

    Attributes
    ----------
    config_filename : str
        name of the configuration file
    section: str
        section of database configuration

    Methods
    -------
    read_db_config()
        Read database configuration file and return a dictionary object.


    """

    def __init__(self, config_filename='./config.ini', section='mysql'):
        """
        Parameters
        ----------
        config_filename: str
            name and path of the configuration file
        section: str, default to mysql
            section of database configuration
        """
        self.config_filename = config_filename
        self.section = section
 
    def read_db_config(self):
        """ Read database configuration file and return a dictionary object.

        Returns
        -------
        Dictionary
            A dictionary of database parameters
        """

        from configparser import ConfigParser

        # create parser and read ini configuration file
        parser = ConfigParser()
        parser.read(self.config_filename)
     
        # get section, default to mysql
        db = {}
        if parser.has_section(self.section):
            items = parser.items(self.section)
            for item in items:
                db[item[0]] = item[1]
        else:
            raise Exception('{0} not found in the {1} file'.\
                            format(self.section, self.config_filename))
     
        return db


