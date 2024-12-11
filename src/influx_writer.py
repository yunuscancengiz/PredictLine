from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv
from src._logger import ProjectLogger
import traceback
import os
import time
from pytz import timezone, UTC
from datetime import datetime

class InfluxWriter:
    load_dotenv()
    TOKEN = os.getenv('MY_INFLUX_TOKEN')
    logger = ProjectLogger(class_name='InfluxWriter').create_logger()

    def __init__(self, token:str, url:str, organization:str):
        self.token = token
        self.url = url
        self.organization = organization
        self.bucket = None

        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.organization)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.logger.info(msg='Influx DB writer api client successfully created!')


    def write_into_influxdb(self, bucket:str, data:dict):
        try:
            print(f"Before conversion: {data['time']}")
            #data['time'] = datetime.fromisoformat(data['time'])
            #data['time'] = data['time'].astimezone(UTC)
            data['time'] = datetime.fromisoformat(data['time'])
            nanosecond_timestamp = int(data['time'].astimezone(UTC).timestamp() * 1e9)
            print(f"After conversion: {data['time']}")
            print(f"Type of time: {type(data['time'])}")

            self.bucket = bucket
            if self.bucket == 'predicted-data' or self.bucket == 'predicted-data-15m':
                point = (
                    Point(measurement_name='prediction')
                    .time(nanosecond_timestamp, WritePrecision.NS)
                    .tag('topic', bucket)
                    .field('PredictedAxialAxisRmsVibration', float(data['PredictedAxialAxisRmsVibration']))
                )

                print(point.to_line_protocol())
            else:
                point = (
                    Point(measurement_name='sensor_data')
                    .time(nanosecond_timestamp, WritePrecision.NS)
                    .tag('topic', bucket)
                    .field('machine', data['machine'])
                    .field('axialAxisRmsVibration', float(data['axialAxisRmsVibration']))
                    .field('radialAxisKurtosis', float(data['radialAxisKurtosis']))
                    .field('radialAxisPeakAcceleration', float(data['radialAxisPeakAcceleration']))
                    .field('radialAxisRmsAcceleration', float(data['radialAxisRmsAcceleration']))
                    .field('radialAxisRmsVibration', float(data['radialAxisRmsVibration']))
                    .field('temperture', float(data['temperature']))
                )
            self.write_api.write(bucket=self.bucket, org=self.organization, record=point, write_precision=WritePrecision.NS)
            #self.logger.info(msg=f'Data uploaded successfully into {self.bucket} named Influx DB bucket.')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while writing into {self.bucket} named Influx DB bucket!')
            self.logger.error(msg=traceback.format_exc())


    def close_connection(self):
        self.client.close()