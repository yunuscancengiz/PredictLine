from confluent_kafka import Consumer, KafkaException
import json
import traceback
from write_to_influx import InfluxDBWriter
from _logger import ProjectLogger

class SimpleConsumer:
    logger = ProjectLogger(class_name='SimpleConsumer').create_logger()
    db_client = InfluxDBWriter(host='localhost', port=8086, dbname='test_db', is_exist=False)
    db_client.create_and_switch_database()

    def __init__(self, topic:str, properties_file:str) -> None:
        self.topic = topic
        self.properties_file = properties_file
        self.consumer_config = {}

        # create consumer config
        self.read_config()

        # create consumer object using config dict
        self.consumer = Consumer(self.consumer_config)


    def main(self):
        try:
            self.consume_messages()
        except Exception as e:
            self.logger.error(msg=f'Exception happened in main function, error: {e}')
            self.logger.error(msg=traceback.format_exc())
        finally:
            self.db_client.disconnect()


    def deserialize_data(self, data):
        return json.loads(data)


    def read_config(self):
        with open(self.properties_file) as fh:
            for line in fh:
                line = line.strip()
                if len(line) != 0 and line[0] != '#':
                    parameter, value = line.strip().split('=', 1)
                    self.consumer_config[parameter] = value
        self.consumer_config['group.id'] = 'python-group-1'
        self.consumer_config['auto.offset.reset'] = 'earliest'


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
                msg = self.deserialize_data(data=msg.value().decode('utf-8'))
                self.logger.info(msg=f'Consumed message: {msg}')

                print(f'\n-------------------\n{type(msg)}\n-------------------\n')
                
                # @TODO: update write_data function with influxdb json format then comment out the next line
                self.db_client.write_data(data=msg)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.logger.error(msg=f'Exception happened when consuming messages, error: {e}')
                self.logger.error(traceback.format_exc())


if __name__ == '__main__':
    simple_consumer = SimpleConsumer(
        topic='test-topic-2',
        properties_file='client.properties'
    )

    simple_consumer.main()