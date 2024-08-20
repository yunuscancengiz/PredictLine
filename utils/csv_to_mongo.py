import pymongo
import pandas as pd


class CsvToMongo:

    def __init__(self, username:str, password:str, cluster_name:str, filename:str) -> None:
        self.username = username
        self.password = password
        self.cluster_name = cluster_name
        self.filename = filename
        self.df = pd.read_csv(self.filename)
        self.data = []


    def main(self):
        self.database_connection()
        self.create_data_list()
        self.write_data(data=self.data)


    def database_connection(self):
        self.cluster = pymongo.MongoClient(f'mongodb+srv://{self.username}:{self.password}@{self.cluster_name.lower()}.g61fksa.mongodb.net/?retryWrites=true&w=majority')
        self.db = self.cluster['Sensor']
        self.collection = self.db['Data']


    def create_data_list(self):
        for index in range(len(self.df)):
            self.data.append(
                {
                    'ts': self.df.loc[index, 'ts'],
                    'device': self.df.loc[index, 'device'],
                    'co': self.df.loc[index, 'co'], 
                    'humidity': self.df.loc[index, 'humidity'],
                    'light': self.df.loc[index, 'light'],
                    'lpg': self.df.loc[index, 'lpg'],
                    'motion': self.df.loc[index, 'motion'],
                    'smoke': self.df.loc[index, 'smoke'],
                    'temp': self.df.loc[index, 'temp']
                }
            )


    def write_data(self, data):
        if type(data) == list:
            try:
                self.collection.insert_many(data)
                print('Data list successfuly wrote to the db!')
            except Exception as e:
                print(f'Exception happened when writing data list to the db, exception: {e}')
        elif type(data) == dict:
            try:
                self.collection.insert_one(data)
                print('Data successfuly wrote to the db!')
            except Exception as e:
                print(f'Exception happened when writing a data dict to the db, exception: {e}')
        else:
            print('Invalid data type!')
        


if __name__ == '__main__':
    db_writer = CsvToMongo(
        username='catchthebalina', 
        password='whaleBot_1104', 
        cluster_name='Whales', 
        filename='mini_data.csv')
    
    db_writer.main()