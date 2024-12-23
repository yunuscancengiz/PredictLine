from confluent_kafka import Producer, KafkaException
import time
import json
import traceback
import pandas as pd
from src._logger import ProjectLogger
from src._create_dataset import DatasetCreator
from dotenv import load_dotenv
import os


class SimpleProducer:
    load_dotenv()
    SERVER_IP = os.getenv('GCP_IP')
    logger = ProjectLogger(class_name='SimpleProducer').create_logger()

    def __init__(self) -> None:
        self.topic = None
        self.data_filename = None
        self.df = None
        self.producer_config = {
            'bootstrap.servers': f'{self.SERVER_IP}:9092'
        }


    def main(self, topic:str, data_filename:str=None, df=None):
        try:
            self.topic = topic
            self.data_filename = data_filename
            self.df = df

            self.prepare_messages()

            # create producer object using config dict
            self.producer  = Producer(self.producer_config)

            self.produce_messages(topic=topic)
        except Exception as e:
            self.logger.error(msg=f'Exception happened in main function, error: {e}')
            self.logger.error(msg=traceback.format_exc())
        except KeyboardInterrupt:
            raise


    def prepare_messages(self):
        if self.df is not None:
            self.messages = self.df
        else:
            if self.data_filename == None:
                # fetch messages
                dataset_creator = DatasetCreator()
                self.data_filename = dataset_creator.main(start='-1d', stop='now()', line='L301', machine='Blower-Pump-1', timeframe='15m')
            
            self.messages = pd.read_csv(self.data_filename)

        if '__time' in self.messages.columns:
            self.messages.rename(columns={'__time': 'time'}, inplace=True)


    def delivery_report(self, err, msg):
        if err is not None:
            self.logger.warning(msg=f'Delivery failed for {msg.key()}, error: {err}')
            return
        #self.logger.info(msg=f'Record: {msg.key()} successfully produced to topic: {msg.topic()} partition: [{msg.partition()}] at offset: {msg.offset()}')


    def serialize_data(self, index:int):
        data = {col: str(self.messages.loc[index, col]) for col in self.messages.columns}
        key = str(int(time.time()))
        value = json.dumps(data).encode(encoding='utf-8')
        return key, value
    

    def produce_messages(self, topic:str):
        self.logger.info(msg=f'Messages are going to produce to the {topic} named topic.')
        for index in range(len(self.messages)):
            try:
                msg_key, msg_value = self.serialize_data(index=index)
                self.producer.produce(key=msg_key, value=msg_value, topic=self.topic, on_delivery=self.delivery_report)
            except BufferError:
                self.producer.poll(0.1)
            except Exception as e:
                self.logger.error(msg=f'Exception while producing message - index: {index}, Err: {e}')
                self.logger.error(msg=traceback.format_exc())
            except KeyboardInterrupt:
                raise
        self.logger.info(msg=f'Messages successfully produced to the {topic} named topic!')
        self.producer.flush()
        time.sleep(3)