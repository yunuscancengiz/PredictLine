from confluent_kafka import Producer, Consumer, KafkaException
from fetch_finance_data import FetchData
import time

class ApprovedProducer:
    def __init__(self, topic:str, symbol:str, properties_file:str, approving_topic:str, approving_properties_file:str) -> None:
        self.topic = topic
        self.symbol = symbol
        self.properties_file = properties_file
        self.approving_topic = approving_topic
        self.approving_properties_file = approving_properties_file
        self.producer_config = {}
        self.consumer_config = {}
        self.messages = []
        self.is_approved = True

        # prepare config file
        self.read_config()

        # create producer object using config dict
        self.producer = Producer(self.producer_config)
        self.consumer = Consumer(self.consumer_config)


    def main(self):
        while True:
            try:
                self.messages = FetchData(symbol=self.symbol).fetch()
                for index in range(len(self.messages)):
                    if self.is_approved:
                        msg_key, msg_value = self.parse_messages(index=index)
                        self.produce_message(message_key=msg_key, message_value=msg_value)
                        self.consume_messages()
                self.messages = []
            except Exception as e:
                print(e)
            except KeyboardInterrupt:
                break

    def main_ex(self):
        while True:
            try:
                if self.is_approved:
                    self.messages = FetchData(symbol=self.symbol).fetch()
                    self.produce_messages()
                    self.messages = []  # @TODO: karşıdan onay gelince sıfırlayacak hale getir
                    self.is_approved = False
                else:
                    self.consume_messages()
            except Exception as e:
                print(e)
            except KeyboardInterrupt:
                break


    def read_config(self):
        with open(self.properties_file) as fh:
            for line in fh:
                line = line.strip()
                if len(line) != 0 and line[0] != '#':
                    parameter, value = line.strip().split('=', 1)
                    self.producer_config[parameter] = value.strip()

        with open(self.approving_properties_file) as fh:
            for line in fh:
                line = line.strip()
                if len(line) != 0 and line[0] != '#':
                    parameter, value = line.strip().split('=', 1)
                    self.consumer_config[parameter] = value.strip()
        self.consumer_config["group.id"] = "python-group-1"
        self.consumer_config["auto.offset.reset"] = "earliest"


    @staticmethod
    def delivery_report(err, msg):
        if err is not None:
            print(f'Delivery failed for {msg.key()}, error: {err}')
            return
        print(f'Record:{msg.key()} successfully produced to topic:{msg.topic()} partition:[{msg.partition()}] at offset:{msg.offset()}')


    def parse_messages(self, index:int):
        key = str(self.messages.loc[index, 'Date'])
        value = f"{str(self.messages.loc[index, 'Open'])}, {str(self.messages.loc[index, 'High'])}, {str(self.messages.loc[index, 'Low'])}, {str(self.messages.loc[index, 'Close'])}"
        return key, value
    

    def consume_messages(self):
        self.consumer.subscribe(topics=[self.approving_topic])
        while True:
            try:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaException._PARTITION_EOF:
                        continue
                    else:
                        print(f'Error: {msg.error()}')
                        break
                msg = f'key: {msg.key()} value: {msg.value()}'
                self.is_approved = True
                return msg
            except Exception as e:
                print(f'Exception happened when consuming messages, error: {e}')
            except KeyboardInterrupt:
                raise


    def produce_message(self, message_key:str, message_value:str):
        try:
            if self.is_approved:
                self.producer.produce(key=message_key, value=message_value, topic=self.topic, on_delivery=self.delivery_report)
                self.is_approved = False
                self.producer.flush()
                #time.sleep(1)
        except BufferError:
            self.producer.poll(0.1)
        except Exception as e:
            print(f'Exception while producing a message, Err: {e}')
        except KeyboardInterrupt:
            raise


    def produce_messages(self):
        for index in range(len(self.messages)):
            try:
                if self.is_approved == True:
                    message_key, message_value = self.parse_messages(index=index)
                    self.producer.produce(key=message_key, value=message_value, topic=self.topic, on_delivery=self.delivery_report)
                    self.is_approved = False    # her mesajda onay mesajı bekler 
            except BufferError:
                self.producer.poll(0.1)
            except Exception as e:
                print(f'Exception while producing message - index: {index}, Err: {e}')
            except KeyboardInterrupt:
                raise
        self.producer.flush()
        time.sleep(3)
        #self.is_approved = False


if __name__ == '__main__':
    appr_producer = ApprovedProducer(
        topic='test-topic-2',
        symbol='AAPL',
        properties_file='client.properties',
        approving_topic='test-topic-3',
        approving_properties_file='client2.properties'
    )

    appr_producer.main()