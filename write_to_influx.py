from influxdb import InfluxDBClient
import traceback
import logging
import pandas as pd


class InfluxDBWriter:
    data = [
        {
            "measurement": "temperature",
            "tags": {
                "location": "office"
            },
            "fields": {
                "value": 23.5
            }
        },
        {
            "measurement": "temperature",
            "tags": {
                "location": "lab"
            },
            "fields": {
                "value": 22.0
            }
        }
    ]

    # create logger object
    logger = logging.getLogger(name='InfluxDBWriter')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def __init__(self, host:str, port:int, dbname:str, is_exist=bool, username:str=None, password:str=None) -> None:
        self.host = host
        self.port = port
        self.dbname = dbname
        self.is_exist = is_exist
        self.username = username
        self.password = password

        # connection
        self.client = self.connect()


    def main(self):
        self.create_and_switch_database()
        self.write_data(data=self.data)
        self.fetch_data(query='SELECT * FROM temperature WHERE "location" = \'office\'')
        #self.delete_database(force=False)
        self.disconnect()


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
        

    def disconnect(self):
        self.client.close()
        self.logger.info(msg='Database connection closed!')


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
                print('delete_database False koşuluna girildi')
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
        returned_data = list(self.client.query(query=query))
        print(returned_data)
        print(type(returned_data))
        
        #return self.client.query(query=query)

if __name__ == '__main__':
    db_client = InfluxDBWriter(
        host='localhost',
        port=8086,
        dbname='test_db',
        is_exist=False
    )

    db_client.main()