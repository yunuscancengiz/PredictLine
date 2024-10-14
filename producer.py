from confluent_kafka import Producer, KafkaException
import time
import json
import traceback
import pandas as pd
from _logger import ProjectLogger
from _create_dataset import DatasetCreator
from dotenv import load_dotenv
import os


class SimpleProducer:
    load_dotenv()
    SERVER_IP = os.getenv('GCP_IP')
    logger = ProjectLogger(class_name='SimpleProducer').create_logger()

    def __init__(self, topic:str) -> None:
        self.topic = topic
        self.producer_config = {
            'bootstrap.servers': f'{self.SERVER_IP}:9092'
        }

        # fetch messages
        dataset_creator = DatasetCreator(start='-1d', stop='now()', line='L301', machine='Blower-Pump-1', timeframe='15m')
        self.data_filename = dataset_creator.main()
        
        self.messages = pd.read_csv(self.data_filename)

        # create producer object using config dict
        self.producer  = Producer(self.producer_config)


    def main(self):
        try:
            self.produce_messages()
        except Exception as e:
            self.logger.error(msg=f'Exception happened in main function, error: {e}')
            self.logger.error(msg=traceback.format_exc())
        except KeyboardInterrupt:
            raise


    def delivery_report(self, err, msg):
        if err is not None:
            self.logger.warning(msg=f'Delivery failed for {msg.key()}, error: {err}')
            return
        self.logger.info(msg=f'Record: {msg.key()} successfuly produced to topic: {msg.topic()} partition: [{msg.partition()}] at offset: {msg.offset()}')


    def serialize_data(self, index:int):
        data = {
            'machine': str(self.messages.loc[index, 'machine']),
            'time': str(self.messages.loc[index, 'time']),
            'axialAxisRmsVibration': str(self.messages.loc[index, 'axialAxisRmsVibration']),
            'radialAxisKurtosis': str(self.messages.loc[index, 'radialAxisKurtosis']),
            'radialAxisPeakAcceleration': str(self.messages.loc[index, 'radialAxisPeakAcceleration']),
            'radialAxisRmsAcceleration': str(self.messages.loc[index, 'radialAxisRmsAcceleration']),
            'radialAxisRmsVibration': str(self.messages.loc[index, 'radialAxisRmsVibration']),
            'temperature': str(self.messages.loc[index, 'temperature'])
        }
        key = str(int(time.time()))
        value = json.dumps(data).encode(encoding='utf-8')
        return key, value
    

    def produce_messages(self):
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
        self.producer.flush()
        time.sleep(3)


if __name__ == '__main__':
    simple_producer = SimpleProducer(topic='raw-data')
    simple_producer.main()