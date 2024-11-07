from _create_dataset import DatasetCreator
from producer import SimpleProducer
from data_processor import DataPreprocessor
from druid_data import DruidDataFetcher
from model import Model
from postgre_db import PostgreClient
import time


class Run:
    def __init__(self) -> None:
        # create dataset
        self.dataset_creator = DatasetCreator(start='-30d', stop='now()', line='L301', machine='Blower-Pump-1', timeframe='1m')
        csv_filename = self.dataset_creator.main()

        # produce raw data to the raw-data topic
        self.raw_data_producer = SimpleProducer(topic='raw-data', data_filename=csv_filename)
        self.raw_data_producer.main()

        # wait for druid db to consume messages from raw-data topic
        time.sleep(15)

        # fetch from druid raw-data
        self.raw_druid_fetcher = DruidDataFetcher(topic='raw-data')
        raw_df = self.raw_druid_fetcher.main()

        # pre-process
        self.processor = DataPreprocessor(df=raw_df)
        processed_df = self.processor.main()

        # produce proccesed data to the processed-data topic
        self.processed_data_producer = SimpleProducer(topic='processed-data', df=processed_df)
        self.processed_data_producer.main()

        # wait for druid db to consume data from proccesed-data topic
        time.sleep(15)

        # fetch processed data from druid's processed data table
        self.processed_druid_fetcher = DruidDataFetcher(topic='processed-data')
        model_df = self.processed_druid_fetcher.main()

        # run model
        self.ml_model = Model(df=model_df)
        model_results = self.ml_model.main()

        # insert model results into postgre db
        self.postgre_client = PostgreClient()
        self.postgre_client.create_table(table_name='results')
        for result in model_results:
            self.postgre_client.insert_data(table_name='results', size=result['size'], model=result['model'], accuracy_score=result['score'])
