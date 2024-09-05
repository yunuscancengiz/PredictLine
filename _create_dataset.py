from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import time


class DatasetCreator:
    load_dotenv()
    BUCKET = os.getenv('INFLUX_BUCKET')
    ORG = os.getenv('INFLUX_ORG')
    TOKEN = os.getenv('INFLUX_TOKEN')
    URL = os.getenv('INFLUX_URL')

    DIRECTORY_NAME = f'{str(int(time.time()))}'
    os.makedirs(os.path.join(os.getcwd(), 'dataset', DIRECTORY_NAME))
    DATASET_PATH = os.path.join(os.getcwd(), 'dataset', DIRECTORY_NAME)

    db_columns = ['', 'result', 'table', '_start', '_stop', '_time', '_value', '_field', '_measurement', 'host', 'line', 'machine', 'name', 'slave_id', 'type']
    df_columns = ['machine', 'time', 'axialAxisRmsVibration', 'radialAxisKurtosis', 'radialAxisPeakAcceleration', 'radialAxisRmsAcceleration', 'radialAxisRmsVibration', 'temperature']
    default_machine_list = ['Blower-Pump-1', 'Blower-Pump-2', 'Blower-Pump-3', 'Blower-Pump-4', 'Vacuum-Pump-2', 'Vacuum-Pump-3', 'Vacuum-Pump-4', 'Vacuum-Pump-5']

    def __init__(self, start:str, stop:str='now()', line:str='L301', machine:str=None, timeframe:str='1m', machine_list:list=None) -> None:
        self.start = start
        self.stop = stop
        self.line = line
        self.machine = machine
        self.timeframe = timeframe
        self.machine_list = machine_list
        self.filename = None

        if self.machine_list is None:
            self.machine_list = self.default_machine_list

        self.client = InfluxDBClient(
            url=self.URL,
            token=self.TOKEN,
            org=self.ORG
        )

        self.query_api = self.client.query_api()


    def main(self):
        for machine in self.machine_list:
            self.machine = machine
            self.filename = self.create_filename()
            self.update_query()
            self.fetch_data()
            self.create_csv()
            time.sleep(15)


    def fetch_data(self):
        rows = self.query_api.query_csv(query=self.query)
        self.df = pd.DataFrame(rows, columns=self.db_columns).iloc[4:, :]
        list_for_df = []
        for time, field, value, machine_name in zip(self.df['_time'], self.df['_field'], self.df['_value'], self.df['machine']):
            list_for_df.append(
                {
                    'time': time,
                    'field': field,
                    'value': value,
                    'machine': machine_name
                }
            )
        self.df = pd.DataFrame(list_for_df)


    def create_csv(self):
        data = {}
        for col in self.df_columns:
            if col == 'machine' or col == 'time':
                data[col] = self.df.loc[self.df['field'] == 'axialAxisRmsVibration'][col].reset_index(drop=True)
            else:
                data[col] = self.df.loc[self.df['field'] == col]['value'].reset_index(drop=True)

        self.df = pd.DataFrame(data)
        print(f'{self.df.describe().T}\n\n')
        self.df.to_csv(f'{self.DATASET_PATH}/{self.filename}', index=False)
        return self.df
    

    def update_query(self):
        self.query = f'''
            from(bucket: "{self.BUCKET}")
            |> range(start: {self.start}, stop: {self.stop})
            |> filter(fn: (r) => r["_measurement"] == "SmartSensor_IC_CHN")
            |> filter(fn: (r) => r["_field"] == "axialAxisRmsVibration" or r["_field"] == "radialAxisKurtosis" or r["_field"] == "radialAxisPeakAcceleration" or r["_field"] == "radialAxisRmsAcceleration" or r["_field"] == "radialAxisRmsVibration" or r["_field"] == "temperature")
            |> filter(fn: (r) => r["host"] == "smart-sensor-china")
            |> filter(fn: (r) => r["line"] == "{self.line}")
            |> filter(fn: (r) => r["machine"] == "{self.machine}")
            |> aggregateWindow(every: {self.timeframe}, fn: last, createEmpty: false)
            |> yield(name: "last")
        '''


    def create_filename(self):
        start = self.start.lstrip('-').rstrip('Z').replace(':', '-').replace(':', '-')
        stop = self.stop.lstrip('-').rstrip('Z').replace(':', '-').replace(':', '-')
        return f'{self.line}-{self.machine}-dataset-{start}-{stop}.csv'


        
if __name__ == '__main__':
    dataset_creator = DatasetCreator(
        start='2024-01-14T00:00:00Z',
        stop='2024-01-16T00:00:00Z',
        machine_list=['Vacuum-Pump-2'],
        line='L302'
    )
    dataset_creator.main()