import psycopg2
import os
from dotenv import load_dotenv
from _logger import ProjectLogger
import traceback


class PostgreClient:
    logger = ProjectLogger(class_name='PostgreClient').create_logger()

    load_dotenv()
    POSTGRE_USERNAME = os.getenv('POSTGRE_USERNAME')
    POSTGRE_PASSWORD = os.getenv('POSTGRE_PASSWORD')
    POSTGRE_HOST = os.getenv('POSTGRE_HOST')
    POSTGRE_PORT = os.getenv('POSTGRE_PORT')
    POSTGRE_DB_NAME = os.getenv('POSTGRE_DB_NAME')

    def __init__(self):
        self.db_client = psycopg2.connect(
            user=self.POSTGRE_USERNAME,
            password=self.POSTGRE_PASSWORD,
            host=self.POSTGRE_HOST,
            port=self.POSTGRE_PORT
        )
        # database=self.POSTGRE_DB_NAME
        self.cursor = self.db_client.cursor()

        print(self.db_client.get_dsn_parameters())


    def create_table(self, table_name:str):
        try:
            query = f'''CREATE TABLE IF NOT EXISTS {table_name}(id SERIAL PRIMARY KEY, size INT NOT NULL, model TEXT NOT NULL, accuracy_score DECIMAL NOT NULL);'''
            self.cursor.execute(query=query)
            self.db_client.commit()
            self.logger.info(msg=f'{table_name} named table already exists or created successfuly.')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while creating {table_name} named table, Error: {e}')
            self.logger.error(msg=f'{str(traceback.format_exc(e))}')


    def insert_data(self, table_name:str, size:int, model:str, accuracy_score:float):
        try:
            query = f'''INSERT INTO {table_name}(size, model, accuracy_score) VALUES({size}, '{model}', {accuracy_score});'''
            self.cursor.execute(query=query)
            self.db_client.commit()
        except Exception as e:
            self.logger.error(msg=f'Exception happened while inserting the data, Error: {e}')
            self.logger.error(msg=f'{str(traceback.format_exc(e))}')


    def fetch_data(self, table_name:str):
        query = f'SELECT * FROM {table_name};'
        self.cursor.execute(query=query)

        results = self.cursor.fetchall()
        for result in results:
            print(result)


    def update_data(self, table_name:str, column_name:str, new_value, _id:int):
        try:
            query = f'UPDATE {table_name} SET {column_name}={new_value} WHERE id={_id};'
            self.cursor.execute(query=query)
            self.db_client.commit()
            self.logger.info(msg='Data updated successfuly!')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while updating the data, Error: {e}')
            self.logger.error(msg=f'{str(traceback.format_exc(e))}')


    def delete_data(self, table_name:str, _id:int):
        try:
            query = f'DELETE FROM {table_name} WHERE id={_id};'
            self.cursor.execute(query=query)
            self.db_client.commit()
            self.logger.info(msg='Data successfuly deleted!')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while deleting the data, Error: {e}')
            self.logger.error(msg=f'{str(traceback.format_exc(e))}')


if __name__ == '__main__':
    postgre_client = PostgreClient()
