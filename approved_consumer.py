from confluent_kafka import Producer, Consumer, KafkaException
from mongodb import MongoDB

class ApprovedConsumer():
    db_username = 'catchthebalina'
    db_password = 'whaleBot_1104'
    db_cluster = 'Whales'
    db_client = MongoDB(username=db_username, password=db_password, cluster_name=db_cluster)

    def __init__(self, topic:str, properties_file:str, approving_properties_file:str, approving_topic:str) -> None:
        self.topic = topic
        self.properties_file = properties_file
        self.approving_properties_file = approving_properties_file
        self.approving_topic = approving_topic
        self.is_wrote = False

        self.consumer_config = {}
        self.producer_config = {}

        # prepare config dicts
        self.read_config()

        # create producer object using config dict
        self.producer = Producer(self.producer_config)
        self.consumer = Consumer(self.consumer_config)


    def main(self):
        while True:
            try:
                if self.is_wrote == False:
                    self.consume_messages()
                    self.is_wrote == True
                else:
                    self.produce_approving_message()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)


    def read_config(self):
        with open(self.properties_file) as fh:
            for line in fh:
                line = line.strip()
                if len(line) != 0 and line[0] != '#':
                    parameter, value = line.strip().split('=', 1)
                    self.consumer_config[parameter] = value
        self.consumer_config["group.id"] = "python-group-1"
        self.consumer_config["auto.offset.reset"] = "earliest"

        with open(self.approving_properties_file) as fh:
            for line in fh:
                line = line.strip()
                if len(line) != 0 and line[0] != '#':
                    parameter, value = line.strip().split('=', 1)
                    self.producer_config[parameter] = value


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
                        print(f'Error: {msg.error()}')
                        break
                msg = f'key: {msg.key()} - value: {msg.value()}'
                print(msg)
                msg = {
                    'date': msg.split(', ')[0],
                    'open': msg.split(', ')[1], 
                    'high': msg.split(', ')[2],
                    'low': msg.split(', ')[3]
                    #'close': msg.split(', ')[4]
                }
                self.db_client.write_data(data=msg)
                self.is_wrote = True
                return msg
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f'Exception happened when consuming messages, error: {e}')     


    @staticmethod
    def delivery_report(err, msg):
        if err is not None:
            print(f'Delivery failed for {msg.key()}, error: {err}')
            return
        print(f'Record:{msg.key()} successfully produced to topic:{msg.topic()} partition:[{msg.partition()}] at offset:{msg.offset()}')


    def produce_approving_message(self):
        while True:
            try:
                if self.is_wrote == True:
                    self.producer.produce(key='True', value='True', topic=self.approving_topic, on_delivery=self.delivery_report)
                    self.is_wrote = False
                    return
            except KeyboardInterrupt:
                raise
            except BufferError:
                self.producer.poll(0.1)
            except Exception as e:
                print(f'Exception happened when producing approve message, error: {e}')         


if __name__ == '__main__':
    appr_consumer = ApprovedConsumer(
        topic='test-topic-2', 
        properties_file='client.properties',
        approving_properties_file='client2.properties',
        approving_topic='test-topic-3')
    
    appr_consumer.main()