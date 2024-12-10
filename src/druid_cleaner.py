import time
import requests
from datetime import timedelta, datetime
import os
from dotenv import load_dotenv
from src._logger import ProjectLogger
import traceback


class DruidCleaner:
    load_dotenv()
    SERVER_IP = os.getenv('GCP_IP')
    COORDINATOR_PORT = 8081
    RETENTION_DAYS = 10
    logger = ProjectLogger(class_name='DruidCleaner').create_logger()


    def __init__(self, datasource:str='processed-data') -> None:
        self.datasource = datasource
        self.DRUID_COORDINATOR_URL = f'http://{self.SERVER_IP}:{self.COORDINATOR_PORT}/druid/coordinator/v1/datasources'


    def main(self):
        try:
            payload = self.create_payload()
            self.clean(payload=payload)
            self.logger.info(msg=f'Old data cleaned from {self.datasource} named datasource successfully.')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while cleaning old data from {self.datasource} named datasource!')
            self.logger.error(msg=traceback.format_exc())


    def create_payload(self):
        max_timestamp = (datetime.utcnow() - timedelta(days=self.RETENTION_DAYS)).isoformat() + 'Z'
        payload = {'interval': f'0000-01-01T00:00:00Z/{max_timestamp}'}
        return payload


    def clean(self, payload):
        response = requests.post(f'{self.DRUID_COORDINATOR_URL}/{self.datasource}/markUnused', json=payload)

        if response.status_code == 200:
            self.logger.info(msg='Old data deletion initiated successfully.')
        else:
            self.logger.warning(msg=f'Failed to initiate data deletion. Status code: {response.status_code}, Response: {response.text}')


if __name__ == '__main__':
    druid_cleaner = DruidCleaner(datasource='raw-data')
    druid_cleaner.main()