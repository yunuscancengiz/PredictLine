from influxdb import InfluxDBClient
import traceback
import logging
import pandas as pd


class InfluxDBWriter:
    data = [
        {
            'test': 'sfjkslfjs', 
            'deneme': '123',
        }, 
        {
            'test': 'uuıfhsnfs',
            'deneme': '39473'
        }
    ]

    def __init__(self, host:str, port:int, dbname:str, is_exist=bool, username:str=None, password:str=None) -> None:
        self.host = host
        self.port = port
        self.dbname = dbname
        self.is_exist = is_exist
        self.username = username
        self.password = password

        # create logger object
        self.logger = logging.Logger(name='InfluxDBWriter')

        # connection
        self.client = self.connect()


    def main(self):
        self.create_and_switch_database()
        self.write_data(data=self.data)
        self.delete_database(force=False)


    def connect(self):
        try:
            if self.username != None and self.password != None:
                self.logger.info(msg='Database client created!')
                return InfluxDBClient(host=self.host, port=self.port, username=self.username, password=self.password)
            else:
                self.logger.info(msg='Database client created!')
                return InfluxDBClient(host=self.host, port=self.port)
        except Exception as e:
            self.logger.error(msg=f'Exception happened when creating database client! Error message: {e}')
            self.logger.error(msg=traceback.format_exc())


    def create_and_switch_database(self):
        if self.is_exist == True:
            try:
                self.client.switch_database(database=self.dbname)
                self.logger.info(msg=f'Database switched into {self.dbname} successfuly!')
            except Exception as e:
                self.logger.error(msg=f'Exception happened when switching into {self.dbname} named database! Error message: {e}')
                self.logger.error(traceback.format_exc())
        else:
            try:
                self.client.create_database(dbname=self.dbname)
                self.logger.info(msg=f'{self.dbname} named database created successfuly!')
            except Exception as e:
                self.logger.error(f'Exception happened when creating {self.dbname} named database! Error message: {e}')
                self.logger.error(traceback.format_exc())
            try:
                self.client.switch_database(database=self.dbname)
                self.logger.info(msg=f'Database switched into {self.dbname} successfuly!')
            except Exception as e:
                self.logger.error(f'Exception happened when switching into {self.dbname} named database! Error message: {e}')
                self.logger.error(traceback.format_exc())


    def write_data(self, data):
        self.client.write_points(points=data)


    def delete_database(self, force:bool):
        try:
            if force == False:
                self.logger.warning(msg=f'{self.dbname} named database is going to be deleted! Do you want to continue? (y/n)')
                choice = input('')
                if choice.lower == 'n':
                    self.logger.info(msg=f'Drop {self.dbname} database process canceled!')
                    return
            self.client.drop_database(dbname=self.dbname)
            self.logger.info(msg=f'{self.dbname} named database dropped successfuly!')
        except Exception as e:
            self.logger.error(msg=f'Exception happened when dropping the {self.dbname} named database! Error message: {e}')
            self.logger.error(traceback.format_exc())

    def fetch_data(self, query:str) -> pd.DataFrame:
        # @TODO: examine the type of the data that query() func returns
        # @TODO: return the data into pandas DataFrame format
        returned_data = self.client.query(query=query)
        print(returned_data)
        print(type(returned_data))
        
        #return self.client.query(query=query)

if __name__ == '__main__':
    db_client = InfluxDBWriter(
        host='34.45.38.113',
        port=8086,
        dbname='test_db',
        is_exist=False
    )

    db_client.main()