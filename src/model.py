import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from src._logger import ProjectLogger
from datetime import datetime
import math
import traceback
import time
from typing import Literal


class RNNModel:
    logger = ProjectLogger(class_name='RNNModel').create_logger()
    thresholds = {
        'axialAxisRmsVibration': 0.1,
        'radialAxisKurtosis': 3,
        'radialAxisPeakAcceleration': 0.05,
        'radialAxisRmsAcceleration': 0.01
    }
    input_columns = ['axialAxisRmsVibration', 'radialAxisKurtosis', 'radialAxisPeakAcceleration', 'radialAxisRmsAcceleration', 'radialAxisRmsVibration', 'temperature', 'is_running']
    target_column = 'axialAxisRmsVibration'
    EPOCHS = 3

    def __init__(self):
        self.df = None
        self.input_steps = None
        self.output_steps = None
        self.window_size = None
        self.model_name = None
        self.model_directory_path = None
        self.train_size = 0.7   # percentage
        self.test_size = 0.2     # percentage
        self.start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


    def main(self, load_best_model:bool, df:pd.DataFrame, input_days:int, output_days:int, interval_minute:int, model_name:str=None):
        self.df = self.preprocess(df=df)
        self.stats = self.calculate_stats(df=self.df, multiplier=3)
        self.window_size = math.floor(len(self.df) / 20)
        self.interval_minute = interval_minute
        self.model_name = model_name
        self.load_best_model = load_best_model

        # delete below if unnecessary
        self.input_steps = int((input_days - output_days) * 24 * (60 / interval_minute))
        self.output_steps = int(output_days * 24 * (60 / interval_minute))

        self.model_directory_path = os.path.join(os.getcwd(), 'models', f'{interval_minute}m')
        if not os.path.exists(self.model_directory_path):
            os.makedirs(self.model_directory_path)
        self.manage_model(job='select')     # select model, make predictions, save best model
        self.manage_model(job='delete')     # delete old models if len(model_files) > 5


    def load_existing_model_and_predict(self, model_name:str):
        try:
            lstm_model = tf.keras.models.load_model(f'{self.model_directory_path}/{model_name}')
            self.logger.info(msg=f'{model_name} named model successfully loaded!')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while loading {model_name} named model!')
            self.logger.error(msg=traceback.format_exc())
            return None, None

        X, y = self.prepare_data(df=self.df, window_size=self.window_size)
        X_train, y_train, X_test, y_test, X_val, y_val = self.split_data(X=X, y=y, train_size=self.train_size, test_size=self.test_size)
        X_train_scaled, X_test_scaled, X_val_scaled, feature_scaler = self.scale_features(X_train=X_train, X_test=X_test, X_val=X_val)
        y_train_scaled, y_test_scaled, y_val_scaled, target_scaler = self.scale_targets(y_train=y_train, y_test=y_test, y_val=y_val)

        predictions = self.predict_future_values(X=X, model=lstm_model, output_steps=self.output_steps, feature_scaler=feature_scaler, target_scaler=target_scaler)
        timestamped_predictions = self.add_time_column_to_predicted_values(predictions=predictions, interval_minute=self.interval_minute)
        breakdown_probability = self.calculate_breakdown_probability(predictions=timestamped_predictions, column=self.target_column)
        results = self.calculate_model_performance(model=lstm_model, X_test_scaled=X_test_scaled, y_test=y_test, target_scaler=target_scaler)
        results['test_MSE'] = 0
        results['test_RMSE'] = 0
        results['breakdown_probability'] = breakdown_probability
        results['timestamp'] = datetime.now().replace(second=0, microsecond=0)
        results['model_name'] = str(self.model_name)
        results = self.convert_numpy_types(data=results)
        self.logger.info(msg=f'results:\n{results}')
        return results, timestamped_predictions


    def train_new_model_and_predict(self):
        X, y = self.prepare_data(df=self.df, window_size=self.window_size)
        X_train, y_train, X_test, y_test, X_val, y_val = self.split_data(X=X, y=y, train_size=self.train_size, test_size=self.test_size)
        X_train_scaled, X_test_scaled, X_val_scaled, feature_scaler = self.scale_features(X_train=X_train, X_test=X_test, X_val=X_val)
        y_train_scaled, y_test_scaled, y_val_scaled, target_scaler = self.scale_targets(y_train=y_train, y_test=y_test, y_val=y_val)

        lstm_model, test_MSE, test_RMSE = self.LSTM_Model(
            X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, X_val_scaled=X_val_scaled,
            y_train_scaled=y_train_scaled, y_test_scaled=y_test_scaled, y_val_scaled=y_val_scaled)
        
        predictions = self.predict_future_values(X=X, model=lstm_model, output_steps=self.output_steps, feature_scaler=feature_scaler, target_scaler=target_scaler)
        timestamped_predictions = self.add_time_column_to_predicted_values(predictions=predictions, interval_minute=self.interval_minute)
        breakdown_probability = self.calculate_breakdown_probability(predictions=timestamped_predictions, column=self.target_column)
        results = self.calculate_model_performance(model=lstm_model, X_test_scaled=X_test_scaled, y_test=y_test, target_scaler=target_scaler)
        results['test_MSE'] = test_MSE
        results['test_RMSE'] = test_RMSE
        results['breakdown_probability'] = breakdown_probability
        results['timestamp'] = datetime.now().replace(second=0, microsecond=0)
        results['model_name'] = str(self.model_name)
        results = self.convert_numpy_types(data=results)
        self.logger.info(msg=f'results:\n{results}')
        return results, timestamped_predictions


    def preprocess(self, df):
        df.index = pd.to_datetime(df['__time'], format='ISO8601')
        df.drop(inplace=True, axis=1, columns=['__time', 'machine'])
        
        df = df[self.input_columns]
        df['is_running'] = 1
        df.loc[
            (df['axialAxisRmsVibration'] < self.thresholds['axialAxisRmsVibration']) & 
            (df['radialAxisKurtosis'] < self.thresholds['radialAxisKurtosis']) & 
            (df['radialAxisPeakAcceleration'] < self.thresholds['radialAxisPeakAcceleration']) & 
            (df['radialAxisRmsAcceleration'] < self.thresholds['radialAxisRmsAcceleration']),
            'is_running'
        ] = 0
        return df
    

    def calculate_stats(self, df, multiplier=3):
        stats = {}
        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (multiplier * IQR)
            upper_bound = Q3 + (multiplier * IQR)

            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            stats[column] = {
                "mean": df[column].mean(),
                "std": df[column].std()
            }
        return stats


    def prepare_data(self, df:pd.DataFrame, window_size:int):
        target_index = df.columns.tolist().index(self.target_column)
        df = df.to_numpy()
        X = []
        y = []

        for index in range(len(df) - window_size):
            X.append([window for window in df[index:index + window_size]])
            y.append(df[index + window_size][target_index])
        return np.array(X), np.array(y)
    

    def split_data(self, X, y, train_size:float=0.7, test_size:float=0.2):
        if train_size + test_size > 1.0:
            self.logger.warning(msg='Train size and test size must sum up to 1 or less.')
            raise ValueError('Train size and test size must sum up to 1 or less.')
        
        train_size = int(len(X) * train_size)
        test_size = int(len(X) * test_size)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:train_size + test_size], y[train_size:train_size + test_size]
        X_val, y_val = X[train_size + test_size:], y[train_size + test_size:]
        return X_train, y_train, X_test, y_test, X_val, y_val
    

    def scale_features(self, X_train, X_test, X_val):
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
        X_val_scaled = feature_scaler.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)
        X_test_scaled = feature_scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
        return X_train_scaled, X_test_scaled, X_val_scaled, feature_scaler
    

    def scale_targets(self, y_train, y_val, y_test):
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
        y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1))
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))
        return y_train_scaled, y_test_scaled, y_val_scaled, target_scaler
    

    def LSTM_Model(self, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, X_val_scaled, y_val_scaled):
        lstm_model = Sequential()
        lstm_model.add(Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
        lstm_model.add(LSTM(100, return_sequences=True))
        lstm_model.add(Dropout(0.3))
        lstm_model.add(LSTM(50))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(8, 'relu'))
        lstm_model.add(Dense(1, 'linear'))
        lstm_model.summary()

        # @TODO:patience argument will be updated as 10
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model_name = f'model_{int(time.time())}_{self.interval_minute}m.keras'
        checkpoint = ModelCheckpoint(f'{self.model_directory_path}/{self.model_name}', save_best_only=True)
        lstm_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
        lstm_model.fit(X_train_scaled, y_train_scaled, validation_data=(X_val_scaled, y_val_scaled), epochs=self.EPOCHS, batch_size=32, callbacks=[checkpoint, early_stopping])
        test_MSE, test_RMSE = lstm_model.evaluate(X_test_scaled, y_test_scaled)
        return lstm_model, test_MSE, test_RMSE
    

    def predict_future_values(self, X, model, output_steps:int, feature_scaler, target_scaler):
        X_scaled = feature_scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
        predictions = []
        last_sequence = X_scaled[-1]

        for _ in range(output_steps):
            pred = model.predict(last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1]))[0][0]
            predictions.append(pred)
            
            new_row = last_sequence[-1].copy()
            for col_idx, col_name in enumerate(self.df.columns):
                if col_name == self.target_column:
                    new_row[col_idx] = pred
                elif col_name == 'is_running':
                    new_row[col_idx] = 1 if pred > self.thresholds[self.target_column] else 0
                else:
                    mean = self.stats[col_name]['mean']
                    std = self.stats[col_name]['std']
                    lower_bound = mean - std
                    upper_bound = mean + std
                    new_row[col_idx] = np.random.uniform(lower_bound, upper_bound)
            last_sequence = np.vstack((last_sequence[1:], new_row))
        predictions_rescaled = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return predictions_rescaled
    

    def add_time_column_to_predicted_values(self, predictions, interval_minute):
        try:
            timestamps = pd.date_range(start=self.start_time, periods=len(predictions), freq=f'{interval_minute}min')
            timestamped_data = pd.DataFrame({
                'time': timestamps,
                'PredictedAxialAxisRmsVibration': predictions
            })
            print(f'\n------------------------\npredicted-data:\n{timestamped_data}')
        except Exception as e:
            self.logger.error(msg='Exception happened while adding time column to the predicted values!')
            self.logger.error(msg=traceback.format_exc())
        return timestamped_data
    

    def calculate_breakdown_probability(self, predictions, column:str):
        try:
            if isinstance(predictions, pd.DataFrame):
                predictions_column = predictions['PredictedAxialAxisRmsVibration']
            else:
                predictions_column = predictions

            threshold = self.thresholds[column]
            number_of_breakdowns = np.sum(predictions_column < threshold)
            breakdown_probability = round(float((number_of_breakdowns * 100) / len(predictions_column)), 2)
            self.logger.info(msg=f'Breakdown probability calculated as {breakdown_probability}%')
        except Exception as e:
            self.logger.error(msg=f'Exception happened while calculating breakdown probability!')
            self.logger.error(msg=traceback.format_exc())
        return breakdown_probability
    

    def calculate_model_performance(self, model, X_test_scaled, y_test, target_scaler):
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}
        return metrics
    

    def convert_numpy_types(self, data):
        if isinstance(data, dict):
            return {key: self.convert_numpy_types(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_numpy_types(item) for item in data]
        elif isinstance(data, np.generic):
            return data.item()
        else:
            return data
        

    def manage_model(self, job:Literal['select', 'delete'], prefix='model_', max_models=5):
        if job not in ['select', 'delete']:
            raise ValueError(f'Invalid job type: {job}. jon must be "select" or "delete".')
        
        suffix = f'_{self.interval_minute}m.keras'
        model_files = [f for f in os.listdir(self.model_directory_path) if f.startswith(prefix) and f.endswith(suffix)]
        model_files.sort(key=lambda x: int(x.split(prefix)[1].split(suffix)[0]), reverse=True)

        if job == 'select':
            if self.load_best_model == True:
                if self.model_name is not None and self.model_name in model_files:
                    self.logger.info(msg=f'Selected model ({self.model_name}) will be used for predictions!')
                    return self.load_existing_model_and_predict(model_name=self.model_name)
                elif len(model_files) > 0:
                    self.model_name = model_files[0]
                    self.logger.info(msg=f'Latest model ({self.model_name}) will be used for predictions!')
                    return self.load_existing_model_and_predict(model_name=self.model_name)
                else:
                    self.logger.warning(msg=f'No saved model found! New model will be trained...')
                    return self.train_new_model_and_predict()
            else:
                self.logger.info(msg='New model will be trained...')
                return self.train_new_model_and_predict()
        elif job == 'delete':
            if len(model_files) > max_models:
                files_to_remove = model_files[5:]
                for f in files_to_remove:
                    os.remove(os.path.join(self.model_directory_path, f))
                self.logger.info(msg=f'Old models deleted! Deleted models: \n{files_to_remove}')



if __name__ == '__main__':
    df_15m = pd.read_csv('dataset/1724929644/dataset-90d.csv')[4::15]
    model = RNNModel()
    model.main(df=df_15m, input_days=90, output_days=10, interval_minute=15)