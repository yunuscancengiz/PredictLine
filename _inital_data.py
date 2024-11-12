from _create_dataset import DatasetCreator
from producer import SimpleProducer
from data_processor import DataPreprocessor
from druid_data import DruidDataFetcher
from datetime import datetime, timedelta
import time


class PrepareInitialData:
    def __init__(self, days_1m:int=14, days_15m:int=90):
        self.starting_date_1m = str((datetime.now() - timedelta(days=days_1m)).isoformat()).split('T')[0] + 'T00:00:00Z'
        self.starting_date_15m = str((datetime.now() - timedelta(days=days_15m)).isoformat()).split('T')[0] + 'T00:00:00Z'
        self.ending_date_1m = str((datetime.now() - timedelta(days=1)).isoformat()).split('T')[0] + 'T23:59:00Z'
        self.ending_date_15m = str((datetime.now() - timedelta(days=1)).isoformat()).split('T')[0] + 'T23:45:00Z'

        self.dataset_creator = DatasetCreator()
        self.producer = SimpleProducer()
        #self.druid_data_fetcher = DruidDataFetcher()


    def main(self):
        filename_1m, filename_15m = self.prepare_datasets()
        self.produce_messages(topic='raw-data', filename=filename_1m) # produce 1m data to raw-data
        self.produce_messages(topic='raw-data-15m', filename=filename_15m)  # produce 15m data raw-data-15m
        time.sleep(15)  # wait for druid's data consuming process



    def prepare_datasets(self):
        filename_1m = self.dataset_creator.main(start=self.starting_date_1m, stop=self.ending_date_1m, line='L301', timeframe='1m')
        filename_15m = self.dataset_creator.main(start=self.starting_date_15m, stop=self.ending_date_15m, line='L301', timeframe='15m')
        return filename_1m, filename_15m


    def produce_messages(self, topic:str, filename:str):
        self.producer.main(topic=topic, data_filename=filename)

        



if __name__ == '__main__':
    data_initializer = PrepareInitialData()
    data_initializer.main()