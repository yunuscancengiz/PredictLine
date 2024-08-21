from confluent_kafka import Producer, KafkaException
from fetch_finance_data import FetchData
import time
import json
import traceback


class SimpleProducer:
    def __init__(self, topic:str, symbol:str, properties_file:str) -> None:
        self.topic = topic
        self.symbol = symbol
        self.properties_file = properties_file
        self.producer_config = {}
        self.messages = []

        # prepare config file
        self.read_config()

        # create producer object using config dict
        self.producer  = Producer(self.producer_config)


    def main(self):
        #while True:
        try:
            self.messages = FetchData(symbol=self.symbol).fetch()
            self.produce_messages()
        except Exception as e:
            print(f'Exception: {e}\n\n{str(traceback.format_exc())}')
        except KeyboardInterrupt:
            raise

    def read_config(self):
        with open(self.properties_file) as fh:
            for line in fh:
                line = line.strip()
                if len(line) != 0 and line[0] != '#':
                    parameter, value = line.strip().split('=', 1)
                    self.producer_config[parameter] = value.strip()


    @staticmethod
    def delivery_report(err, msg):
        if err is not None:
            print(f'Delivery failed for {msg.key()}, error: {err}')
            return
        print(f'Record:{msg.key()} successfully produced to topic:{msg.topic()} partition:[{msg.partition()}] at offset:{msg.offset()}')


    def serialize_data(self, index:int):
        key = str(int(time.time()))
        value = json.dumps(self.messages[index]).encode(encoding='utf-8')
        return key, value


    def parse_messages(self, index:int):
        key = str(self.messages.loc[index, 'Date'])
        value = f"{str(self.messages.loc[index, 'Open'])}, {str(self.messages.loc[index, 'High'])}, {str(self.messages.loc[index, 'Low'])}, {str(self.messages.loc[index, 'Close'])}"
        return key, value
    

    def produce_messages(self):
        for index in range(len(self.messages)):
            try:
                msg_key, msg_value = self.parse_messages(index=index)
                self.producer.produce(key=msg_key, value=msg_value, topic=self.topic, on_delivery=self.delivery_report)
            except BufferError:
                self.producer.poll(0.1)
            except Exception as e:
                print(f'Exception while producing message - index: {index}, Err: {e}\n\n{str(traceback.format_exc())}')
            except KeyboardInterrupt:
                raise
        self.producer.flush()
        time.sleep(3)


if __name__ == '__main__':
    simple_producer = SimpleProducer(
        topic='test-topic-2',
        symbol='AAPL',
        properties_file='client.properties'
    )

    simple_producer.main()