import influxdb_client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv
import os
import time


load_dotenv()

token = os.getenv('MY_INFLUX_TOKEN2')
org = "NISO"
url = "http://34.41.194.149:8086"
bucket="sensor-data"

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)


# WRITE
'''write_api = client.write_api(write_options=SYNCHRONOUS)
   
for value in range(5):
  point = (
    Point("measurement1")
    .tag("tagname1", "tagvalue1")
    .field("field1", value)
  )
  write_api.write(bucket=bucket, org="NISO", record=point)
  time.sleep(1) # separate points by 1 second'''


# QUERY
query_api = client.query_api()

query = """from(bucket: "sensor-data")
 |> range(start: -10m)
 |> filter(fn: (r) => r._measurement == "measurement1")"""
tables = query_api.query(query, org="NISO")

for table in tables:
  for record in table.records:
    print(record)


# MEAN FUNC
'''query_api = client.query_api()

query = """from(bucket: "sensor-data")
  |> range(start: -10m)
  |> filter(fn: (r) => r._measurement == "measurement1")
  |> mean()"""
tables = query_api.query(query, org="NISO")

for table in tables:
    for record in table.records:
        print(record)'''