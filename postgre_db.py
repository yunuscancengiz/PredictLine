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
        self.cursor = self.db_client.cursor()


    def create_table(self, table_name:str):
        try:
            query = f'''
                CREATE TABLE IF NOT EXISTS {table_name}(
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    model_name TEXT NOT NULL,
                    lstm_loss FLOAT NOT NULL,
                    MAE FLOAT NOT NULL,
                    MSE FLOAT NOT NULL,
                    RMSE FLOAT NOT NULL,
                    MAPE FLOAT NOT NULL,
                    R2 FLOAT NOT NULL,
                    breakdown_probability FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            '''
            
            # self.db_client.commit()
            with self.db_client:
                self.cursor.execute(query=query)
                self.logger.info(msg=f'{table_name} named table already exists or created successfully.')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while creating {table_name} named table, Error: {e}')
            self.logger.error(msg=traceback.format_exc())


    def insert_data(self, table_name:str, results:dict):
        try:
            query = f'''
                INSERT INTO {table_name} (
                    timestamp, model_name, lstm_loss, MAE, MSE, RMSE, MAPE, R2, breakdown_probability
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            '''
            values = (
                results["timestamp"],
                results["model_name"],
                results["lstm_loss"],
                results["MAE"],
                results["MSE"],
                results["RMSE"],
                results["MAPE"],
                results["R2"],
                results["breakdown_probability"]
            )

            with self.db_client:
                self.cursor.execute(query, values)
                self.logger.info(msg=f'Data successfully inserted into {table_name}.')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while inserting the data into {table_name}, Error: {e}')
            self.logger.error(msg=traceback.format_exc())


    def fetch_data(self, table_name:str):
        query = f'SELECT * FROM {table_name};'
        self.cursor.execute(query=query)
        results = self.cursor.fetchall()
        return results


    def update_data(self, table_name:str, column_name:str, new_value, _id:int):
        try:
            query = f'UPDATE {table_name} SET {column_name}={new_value} WHERE id={_id};'
            self.cursor.execute(query=query)
            self.db_client.commit()
            self.logger.info(msg='Data updated successfully!')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while updating the data, Error: {e}')
            self.logger.error(msg=traceback.format_exc())


    def delete_data(self, table_name:str, _id:int):
        try:
            query = f'DELETE FROM {table_name} WHERE id={_id};'
            self.cursor.execute(query=query)
            self.db_client.commit()
            self.logger.info(msg='Data successfully deleted!')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while deleting the data, Error: {e}')
            self.logger.error(msg=traceback.format_exc())


if __name__ == '__main__':
    postgre_client = PostgreClient()
