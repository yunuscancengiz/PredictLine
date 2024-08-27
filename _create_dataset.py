from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

class CreateDataset:
    load_dotenv()
    BUCKET = os.getenv('INFLUX_BUCKET')
    ORG = os.getenv('INFLUX_ORG')
    TOKEN = os.getenv('INFLUX_TOKEN')
    URL = os.getenv('INFLUX_URL')

    db_columns = ['', 'result', 'table', '_start', '_stop', '_time', '_value', '_field', '_measurement', 'host', 'line', 'machine', 'name', 'slave_id', 'type']

    def __init__(self, start:str, stop:str='now()', machine:str='Blower-Pump-1', timeframe:str='1m') -> None:
        self.start = start
        self.stop = stop
        self.machine = machine
        self.timeframe = timeframe

        self.client = InfluxDBClient(
            url=self.URL,
            token=self.TOKEN,
            org=self.ORG
        )

        self.query_api = self.client.query_api()

        self.query = f'''
            from(bucket: "{self.BUCKET}")
            |> range(start: -{self.start}, stop: {self.stop})
            |> filter(fn: (r) => r["_measurement"] == "SmartSensor_IC_CHN")
            |> filter(fn: (r) => r["_field"] == "axialAxisRmsVibration" or r["_field"] == "radialAxisKurtosis" or r["_field"] == "radialAxisPeakAcceleration" or r["_field"] == "radialAxisRmsAcceleration" or r["_field"] == "radialAxisRmsVibration" or r["_field"] == "temperature")
            |> filter(fn: (r) => r["host"] == "smart-sensor-china")
            |> filter(fn: (r) => r["line"] == "L301")
            |> filter(fn: (r) => r["machine"] == "{self.machine}")
            |> aggregateWindow(every: {self.timeframe}, fn: last, createEmpty: false)
            |> yield(name: "last")
        '''


    def fetch_data(self):
        rows = self.query_api.query_csv(query=self.query)
        self.df = pd.DataFrame(rows, columns=self.columns).iloc[4:, :]
        list_for_df = []
        for time, field, value, machine in zip(self.df['_time'], self.df['_field'], self.df['_value'], self.df['machine']):
            list_for_df.append(
                {
                    'time': time,
                    'field': field,
                    'value': value,
                    'machine': machine
                }
            )
        self.df = pd.DataFrame(list_for_df)


    def create_csv(self):
        pass
