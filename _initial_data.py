from _create_dataset import DatasetCreator
from producer import SimpleProducer
from data_processor import DataPreprocessor
from druid_data import DruidDataFetcher
from datetime import datetime, timedelta
import time
import pandas as pd
from _logger import ProjectLogger


class PrepareInitialData:
    logger = ProjectLogger(class_name='PrepareInitialData').create_logger()

    def __init__(self, days_1m:int=14, days_15m:int=90):
        self.starting_date_1m = str((datetime.now() - timedelta(days=days_1m)).isoformat()).split('T')[0] + 'T00:00:00Z'
        self.starting_date_15m = str((datetime.now() - timedelta(days=days_15m)).isoformat()).split('T')[0] + 'T00:00:00Z'
        self.ending_date_1m = str((datetime.now() - timedelta(days=1)).isoformat()).split('T')[0] + 'T23:59:00Z'
        self.ending_date_15m = str((datetime.now() - timedelta(days=1)).isoformat()).split('T')[0] + 'T23:45:00Z'

        self.dataset_creator = DatasetCreator()
        self.producer = SimpleProducer()
        self.druid_data_fetcher = DruidDataFetcher()
        self.preprocessor = DataPreprocessor()


    def main(self):
        raw_df_1m = self.prepare_datasets(start=self.starting_date_1m, stop=self.ending_date_1m, timeframe='1m')
        raw_df_15m = self.prepare_datasets(start=self.starting_date_15m, stop=self.ending_date_15m, timeframe='15m')
        self.produce_messages(topic='raw-data', df=raw_df_1m) # produce 1m data to raw-data
        self.produce_messages(topic='raw-data-15m', df=raw_df_15m)  # produce 15m data raw-data-15m
        time.sleep(15)  # wait for druid's data consuming process
        input('After introducing the Kafka topics to Druid, press Enter.')
        df_1m = self.fecth_druid_data(topic='raw-data')     # fetch 1m data from druid raw-data topic
        df_15m = self.fecth_druid_data(topic='raw-data-15m') # fetch 15m data from druid raw-data-15m topic
        processed_df_1m = self.process_data(df=df_1m)   # process 1m data
        processed_df_15m = self.process_data(df=df_15m) # process 15m data
        self.produce_messages(topic='processed-data', df=processed_df_1m)   # send processed 1m data to processed-data 
        self.produce_messages(topic='processed-data-15m', df=processed_df_15m)  # send processed 15m data to p-d-15m


    def prepare_datasets(self, start:str, stop:str, timeframe:str):
        filename = self.dataset_creator.main(start=start, stop=stop, line='L301', timeframe=timeframe, machine='Blower-Pump-1')
        return filename


    def produce_messages(self, topic:str, filename=None, df=None):
        if filename != None:
            self.producer.main(topic=topic, data_filename=filename)
        elif df is not None:
            self.producer.main(topic=topic, df=df)
        else:
            self.logger.warning(msg='Datasource is not valid!')


    def fecth_druid_data(self, topic:str) -> pd.DataFrame:
        return self.druid_data_fetcher.main(topic=topic)
    

    def process_data(self, df:pd.DataFrame):
        return self.preprocessor.main(df=df)


if __name__ == '__main__':
    data_initializer = PrepareInitialData()
    data_initializer.main()