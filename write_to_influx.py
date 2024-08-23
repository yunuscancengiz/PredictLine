from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client .client.write_api import SYNCHRONOUS


class RawDataWriter:
    def __init__(self) -> None:
        self.url = 'http://34.45.38.113:8086'
        self.token = '222QEIh4v8r79e3ch2l6KxtHLLm1Uu4jFixG8oEKeiOE0QJdGI9KnfXbbH5kCXKz1yUsEP4MBUxHo7GYjPPQjQ=='
        self.org = 'N'
        self.bucket = 'test_bucket'

        # create db client
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)

        # run app
        self.main()


    def main(self):
        self.create_bucket()
        self.write_into_bucket()
        self.query_data()


    def create_bucket(self):
        buckets_api = self.client.buckets_api()
        buckets_api.create_bucket(bucket_name=self.bucket, org=self.org)


    def write_into_bucket(self):
        write_api = self.client.write_api(write_options=SYNCHRONOUS)
        data = [
            Point('temperature').tag('location', 'office').field('value', 23.5).time('2024-07-01T00:00:00Z', WritePrecision.NS),
            Point('temperature').tag('location', 'lab').field('value', 38.5).time('2024-07-01T00:00:00Z', WritePrecision.NS),
        ]

        write_api.write(bucket=self.bucket, org=self.org, record=data)


    def query_data(self):
        query_api = self.client.query_api()
        query = f'from(bucket: {self.bucket}) |> range(start: -10y) |> filter(fn: (r) => r._measurement == "temperature" and r.location == "office")'
        results = query_api.query(org=self.org, query=query)

        for table in results:
            for record in table.records:
                print(record.values)


    def delete_data(self):
        start = '2023-01-01T00:00:00Z'
        stop = '2023-01-01T02:00:00Z'
        delete_api = self.client.delete_api()
        delete_api.delete(start=start, stop=stop, predicate='_measurement="temperature" AND location="lab"', bucket=self.bucket, org=self.org)