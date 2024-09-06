import random
from postgre_db import PostgreClient


class RandomModelResults:
    models = ['LSTM', 'Logistic Regression', 'Decision Tree', 'SVM']
    sizes = [1000, 5000, 10000, 20000, 50000, 100000]

    postgre_client = PostgreClient()

    def __init__(self) -> None:
        self.results = []

        # run app
        print('1')
        self.postgre_client.create_table(table_name='results')
        print('2')
        self.generate_random_results()
        print('3')
        self.insert_results_to_db()
        print('4')
        self.check_data()
        print('5')
        

    def generate_random_results(self):
        for size in self.sizes:
            for model in self.models:
                score = round(random.uniform(a=25, b=100), 2)
                result = {
                    'size': size,
                    'model': model,
                    'score': score
                }
                self.results.append(result)


    def insert_results_to_db(self):
        for result in self.results:
            self.postgre_client.insert_data(table_name='results', size=result['size'], model=result['model'], accuracy_score=result['score'])


    def check_data(self):
        self.postgre_client.fetch_data(table_name='results')

if __name__ == '__main__':
    random_result_generator = RandomModelResults()