from confluent_kafka import Consumer, KafkaException
import json
import traceback
from _logger import ProjectLogger
from dotenv import load_dotenv
import os


class SimpleConsumer:
    load_dotenv()
    SERVER_IP = os.getenv('GCP_IP')
    logger = ProjectLogger(class_name='SimpleConsumer').create_logger()

    def __init__(self, topic:str) -> None:
        self.topic = topic

        # create consumer config
        self.consumer_config = {
            'bootstrap.servers': f'{self.SERVER_IP}:9092',
            'group.id': 'my-group',
            'auto.offset.reset': 'earliest'
        }

        # create consumer object using config dict
        self.consumer = Consumer(self.consumer_config)


    def main(self):
        try:
            self.consume_messages()
        except Exception as e:
            self.logger.error(msg=f'Exception happened in main function, error: {e}')
            self.logger.error(msg=traceback.format_exc())
        finally:
            pass
            #self.db_client.disconnect()


    def deserialize_data(self, data):
        return json.loads(data)


    def consume_messages(self):
        self.consumer.subscribe(topics=[self.topic])
        while True:
            try:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error() is not None:
                    if msg.error().code() == KafkaException._PARTITION_EOF:
                        continue
                    else:
                        self.logger.error(msg=f'Error: {msg.error()}')
                        break
                msg = self.deserialize_data(data=msg.value())
                self.logger.info(msg=f'Consumed message: {msg}')
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.logger.error(msg=f'Exception happened when consuming messages, error: {e}')
                self.logger.error(traceback.format_exc())


if __name__ == '__main__':
    simple_consumer = SimpleConsumer(topic='raw-data')
    simple_consumer.main()