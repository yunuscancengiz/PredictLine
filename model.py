import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from _logger import ProjectLogger
import traceback


class RNNModel:
    logger = ProjectLogger(class_name='RNNModel').create_logger()
    thresholds = {
        'axialAxisRmsVibration': 0.1,
        'radialAxisKurtosis': 3,
        'radialAxisPeakAcceleration': 0.05,
        'radialAxisRmsAcceleration': 0.01
    }


    def __init__(self):
        self.df = None
        self.input_steps = None
        self.output_steps = None
        self.start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.scaler = MinMaxScaler()
        self.input_columns = ['axialAxisRmsVibration', 'radialAxisKurtosis', 'radialAxisPeakAcceleration', 'radialAxisRmsAcceleration', 'radialAxisRmsVibration', 'temperature', 'is_running']
        self.target_index = self.input_columns.index('axialAxisRmsVibration')


    def main(self, df:pd.DataFrame, input_days:int, output_days:int, interval_minute:int):
        self.df = df
        self.scaled_df = self.scaler.fit_transform(self.df[self.input_columns])     # normalize data
        self.input_steps, self.output_steps = (input_days * 24 * (60 / interval_minute)), (output_days * 24 * (60 / interval_minute))

        X, y = self.prepare_data(df=self.scaled_df)
        X_train, X_test, y_train, y_test = self.split_and_reshape(X=X, y=y)
        X_test, y_test, y_pred, lstm_loss = self.LSTM_Model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        breakdown_probability = self.calculate_breakdown_probability(y_pred=y_pred)
        results = self.calculate_model_performance(y_test=y_test, y_pred=y_pred)
        results['lstm loss'] = lstm_loss
        results['breakdown probability'] = breakdown_probability
        results['timestamp'] = datetime.now().replace(second=0, microsecond=0)
        predicted_data = self.add_time_column_to_predicted_values(y_pred=y_pred, interval_minute=interval_minute)
        return results, predicted_data


    def preprocess(self, df):
        # model takes processed data from druid. So this function is not going to be used in the main func.
        df['is_running'] = 1
        df.loc[
            (df['axialAxisRmsVibration'] < self.thresholds['axialAxisRmsVibration']) & 
            (df['radialAxisKurtosis'] < self.thresholds['radialAxisKurtosis']) & 
            (df['radialAxisPeakAcceleration'] < self.thresholds['radialAxisPeakAcceleration']) & 
            (df['radialAxisRmsAcceleration'] < self.thresholds['radialAxisRmsAcceleration']),
            'is_running'
        ] = 0
        return df


    def prepare_data(self, df:pd.DataFrame):
        data = df[:, self.target_index].reshape(-1, 1)
        X, y = [], []

        for i in range(len(data) - self.input_steps - self.output_steps + 1):
            X.append(df[i:i + self.input_steps])  # Use all columns for X
            y.append(data[i + self.input_steps:i + self.input_steps + self.output_steps])  # Only target column for y

        return np.array(X), np.array(y)
    

    def split_and_reshape(self, X, y):
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Reshape input data to 3D for LSTM[samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(self.input_columns)))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(self.input_columns)))
        return X_train, X_test, y_train, y_test
    

    def LSTM_Model(self, X_train, X_test, y_train, y_test):
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(y_train.shape[1])  # Output layer matching the output_steps
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train LSTM model
        lstm_model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        lstm_loss = lstm_model.evaluate(X_test, y_test)
        y_pred = lstm_model.predict(X_test)
        return X_test, y_test, y_pred, lstm_loss
    

    def calculate_breakdown_probability(self, y_pred, column:str):
        try:
            if len(y_pred) == 0:
                self.logger.warning(msg='Breakdown probability can not calculated because y_test is an empty list!')
                return 0
        
            threshold = self.thresholds[column]
            number_of_breakdowns = sum(value < threshold for value in y_pred)
            breakdown_probability = round(float((number_of_breakdowns * 100) / len(y_pred)), 2)
            self.logger.info(msg=f'Breakdown probability calculated as {breakdown_probability}%')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while calculating breakdown probability!')
            self.logger.error(msg=traceback.format_exc())
        return breakdown_probability
    

    def add_time_column_to_predicted_values(self, y_pred, interval_minute):
        try:
            timestamps = pd.date_range(start=self.start_time, periods=len(y_pred), freq=f'{interval_minute}T')
            predicted_data = pd.DataFrame({
                'time': timestamps,
                'PredictedAxialAxisRmsVibration': y_pred
            })
        except Exception as e:
            self.logger.error(msg='Exception happened while adding time column to the predicted values!')
            self.logger.error(msg=traceback.format_exc())
        return predicted_data
    

    def calculate_model_performance(self, y_test, y_pred):
        y_pred_binary = np.round(y_pred).flatten()
        y_test_binary = np.round(y_test).flatten()

        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        f1 = f1_score(y_test_binary, y_pred_binary)
        precision = precision_score(y_test_binary, y_pred_binary)
        recall = recall_score(y_test_binary, y_pred_binary)

        metrics = {
            'accuracy score': accuracy,
            'f1 score': f1,
            'precision': precision,
            'recall': recall
        }
        return metrics

    

if __name__ == '__main__':
    model = RNNModel(filename='dataset/1724929644/dataset-60d-30d.csv')
    model.main()