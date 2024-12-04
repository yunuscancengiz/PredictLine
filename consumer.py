from confluent_kafka import Consumer, KafkaException, KafkaError
import json
import traceback
from _logger import ProjectLogger
from dotenv import load_dotenv
from influx_writer import InfluxWriter
import os


class SimpleConsumer:
    load_dotenv()
    TOKEN = os.getenv('MY_INFLUX_TOKEN')
    INFLUX_URL = os.getenv('MY_INFLUX_URL')
    INFLUX_ORG = os.getenv('MY_INFLUX_ORG')

    SERVER_IP = os.getenv('GCP_IP')
    logger = ProjectLogger(class_name='SimpleConsumer').create_logger()

    def __init__(self) -> None:
        self.topic = None
        self.influx_bucket = None
        self.influx_db_client = InfluxWriter(token=self.TOKEN, url=self.INFLUX_URL, organization=self.INFLUX_ORG)

        # create consumer config
        self.consumer_config = {
            'bootstrap.servers': f'{self.SERVER_IP}:9092',
            'group.id': 'my-group',
            'auto.offset.reset': 'earliest'
        }

        # create consumer object using config dict
        self.consumer = Consumer(self.consumer_config)


    def main(self, topics:list, influx_bucket:str):
        try:
            self.topics = topics
            self.influx_bucket = influx_bucket
            self.consume_messages()
        except Exception as e:
            self.logger.error(msg=f'Exception happened in main function, error: {e}')
            self.logger.error(msg=traceback.format_exc())
        finally:
            self.influx_db_client.close_connection()


    def deserialize_data(self, data):
        return json.loads(data)


    def consume_messages(self):
        self.consumer.subscribe(topics=[self.topics])
        while True:
            try:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error() is not None:
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        self.logger.error(msg=f'Error: {msg.error()}')
                        break
                msg = self.deserialize_data(data=msg.value())
                #self.logger.info(msg=f'Consumed message: {msg}')
                self.influx_db_client.write_into_influxdb(bucket=self.influx_bucket, data=msg)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.logger.error(msg=f'Exception happened when consuming messages, error: {e}')
                self.logger.error(traceback.format_exc())


if __name__ == '__main__':
    simple_consumer = SimpleConsumer(influx_bucket='test')
    simple_consumer.main(topics=['raw-data', 'raw-data-15m', 'predicted-data', 'predicted-data-15m'])