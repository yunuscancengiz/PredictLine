import requests
import json
import os
from dotenv import load_dotenv
import traceback
from _logger import ProjectLogger
import pandas as pd


class DruidDataFetcher:
    load_dotenv()
    SERVER_IP = os.getenv('GCP_IP')
    PORT = 8888

    logger = ProjectLogger(class_name='DruidDataFetcher').create_logger()

    def __init__(self):
        self.topic = None
        self.url = f'http://{self.SERVER_IP}:{self.PORT}/druid/v2/sql'


    def main(self, topic:str):
        try:
            self.topic = topic

            data = self.fetch()
            df = self.convert_into_df(data=data)
            return df
        except Exception as e:
            self.logger.error(msg=f'Exception happened while fetching data from {self.topic} named table!')
            self.logger.error(msg=traceback.format_exc())


    def fetch(self):
        query = f'SELECT * FROM "{self.topic}"'
        payload = json.dumps({'query': query})
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.url, headers=headers, data=payload)
        if response.status_code == 200:
            self.logger.info(msg=f'Data successfully fetched from {self.topic} named table!')
            data = response.json()
            return data

        else:
            self.logger.warning(msg=f'Exception happened while fetching data from {self.topic} named table. That might cause an error.')


    def convert_into_df(self, data:list):
        df = pd.DataFrame(data=data)
        df = df.drop(['kafka.timestamp', 'kafka.key', 'kafka.topic'], axis=1)
        self.logger.info(msg='Data converted into a dataframe...')
        return df


if __name__ == '__main__':
    druid_fetcher = DruidDataFetcher(topic='raw-data')
    druid_fetcher.main()