from _create_dataset import DatasetCreator
from producer import SimpleProducer
from data_processor import DataPreprocessor
from druid_data import DruidDataFetcher
from model import RNNModel
from postgre_db import PostgreClient
import time
from datetime import datetime, timedelta


class RunPipeline:
    def __init__(self):
        self.dataset_creator = DatasetCreator()
        self.producer = SimpleProducer()
        self.druid_fetcher = DruidDataFetcher()
        self.preprocesser = DataPreprocessor()
        self.lstm_model = RNNModel()
        self.postgre_client = PostgreClient()

        self.starting_date = None
        self.ending_date = None


    def run(self):
        while True:
            now = datetime.now()
            if now.hour == 3 and now.minute == 22:
                self.pipeline()

            # sleep until next midnight
            tomorrow = datetime.now() + timedelta(days=1)
            next_midnight = datetime.combine(tomorrow.date(), datetime.min.time())
            sleep_seconds = (next_midnight - datetime.now()).total_seconds()
            print(sleep_seconds)
            time.sleep(sleep_seconds)


    def pipeline(self):
        # calculate starting and ending dates
        self.starting_date = str((datetime.now() - timedelta(days=1)).isoformat()).split('T')[0] + 'T00:00:00Z'
        self.ending_date = str((datetime.now() - timedelta(days=1)).isoformat()).split('T')[0] + 'T23:59:00Z'

        # create dataset
        filename_1m = self.dataset_creator.main(start=self.starting_date, stop=self.ending_date, line='L301', timeframe='1m', machine='Blower-Pump-1')
        filename_15m = self.dataset_creator.main(start=self.starting_date, stop=self.ending_date, line='L301', timeframe='15m', machine='Blower-Pump-1')

        # produce raw data
        self.producer.main(topic='raw-data', data_filename=filename_1m)
        self.producer.main(topic='raw-data-15m', data_filename=filename_15m)

        # fetch raw data from druid
        time.sleep(15)  # wait for druid to consume the raw data from kafka topics
        df_1m = self.druid_fetcher.main(topic='raw-data')
        df_15m = self.druid_fetcher.main(topic='raw-data-15m')

        # pre-process data
        processed_df_1m = self.preprocesser.main(df=df_1m)
        processed_df_15m = self.preprocesser.main(df=df_15m)

        # produce processed data
        self.producer.main(topic='processed-data', df=processed_df_1m)
        self.producer.main(topic='processed-data-15m', df=processed_df_15m)

        # fetch processed data from druid
        time.sleep(15)  # wait for druid to consume the processed data from kafka topics
        df_1m = self.druid_fetcher.main(topic='processed-data')
        df_15m = self.druid_fetcher.main(topic='processed-data-15m')

        # run lstm model
        breakdown_probability_1m = self.lstm_model.main(df=df_1m, input_days=14, output_days=2, interval_minute=1)
        breakdown_probability_15m = self.lstm_model.main(df=df_15m, input_days=90, output_days=10, interval_minute=15)

        # insert model results into postgre db
        self.postgre_client.create_table(table_name='results')
        for result in [breakdown_probability_1m, breakdown_probability_15m]:
            self.postgre_client.insert_data(table_name='results', size=result['size'], model=result['model'], accuracy_score=result['score'])


if __name__ == '__main__':
    run_pipeline = RunPipeline()
    run_pipeline.run()