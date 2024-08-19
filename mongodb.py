import pymongo


class MongoDB:
    def __init__(self, username:str, password:str, cluster_name:str) -> None:
        self.username = username
        self.password = password
        self.cluster_name = cluster_name
        self.database_connection()


    def database_connection(self):
        self.cluster = pymongo.MongoClient(f'mongodb+srv://{self.username}:{self.password}@{self.cluster_name.lower()}.g61fksa.mongodb.net/?retryWrites=true&w=majority')
        self.db = self.cluster['Yfinance']
        self.collection = self.db['finance_data']


    def write_data(self, data):
        if type(data) == dict:
            self.collection.insert_one(data)
        elif type(data) == list:
            self.collection.insert_many(data)
        else:
            print(f'Unsupported data format! Data must be in dict or list format.')


    def read_data(self) -> list:
        data = []
        for data_point in self.collection.find():
            data.append(data_point)
        return data



if __name__ == '__main__':
    mng = MongoDB(username='catchthebalina', password='whaleBot_1104', cluster_name='Whales')
    data = [
        {
        'date': '2023-08-12',
        'open': '234.54',
        'high': '267.58',
        'low': '223.76',
        'close': '246.79'
        },
        {
        'date': '2023-08-12',
        'open': '234.54',
        'high': '267.58',
        'low': '223.76',
        'close': '246.79'
        },
        {
        'date': '2023-08-12',
        'open': '234.54',
        'high': '267.58',
        'low': '223.76',
        'close': '246.79'
        },
        {
        'date': '2023-08-12',
        'open': '234.54',
        'high': '267.58',
        'low': '223.76',
        'close': '246.79'
        },
        {
        'date': '2023-08-12',
        'open': '234.54',
        'high': '267.58',
        'low': '223.76',
        'close': '246.79'
        },
        {
        'date': '2023-08-12',
        'open': '234.54',
        'high': '267.58',
        'low': '223.76',
        'close': '246.79'
        }
    ]

    mng.write_data(data=data)
    data = mng.read_data()

    for i in data:
        print(i)
    
