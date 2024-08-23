from influxdb import InfluxDBClient


class InfluxDB:
    def __init__(self, db_name:str) -> None:
        try:
            self.db_name = db_name
            self.client = InfluxDBClient(host='34.45.38.113', port=8086)
            self.create_database(self.db_name)
            self.client.switch_database(self.db_name)
        except Exception as e:
            print(e)
        finally:
            self.client.close()


    def write_data(self, data):
        self.client.write_points(data)


    def delete_database(self):
        self.client.drop_database(dbname=self.db_name)


    def fetch_data(self, query:str):
        return self.client.query(query)