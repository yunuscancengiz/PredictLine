import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from _logger import ProjectLogger
from datetime import datetime
import math
import traceback


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

    def __init__(self):
        self.df = None
        self.input_steps = None
        self.output_steps = None
        self.window_size = None
        self.train_size = 0.7   # percentage
        self.test_size = 0.2     # percentage
        self.start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


    def main(self, df:pd.DataFrame, input_days:int, output_days:int, interval_minute:int):
        self.df = self.preprocess(df=df)
        self.stats = self.calculate_stats(df=self.df, multiplier=3)
        self.window_size = math.floor(len(self.df) / 20)
        self.interval_minute = interval_minute

        # delete below if unnecessary
        self.input_steps = int((input_days - output_days) * 24 * (60 / interval_minute))
        self.output_steps = int(output_days * 24 * (60 / interval_minute))
        

        X, y = self.prepare_data(df=self.df, window_size=self.window_size)
        X_train, y_train, X_test, y_test, X_val, y_val = self.split_data(X=X, y=y, train_size=self.train_size, test_size=self.test_size)
        X_train_scaled, X_test_scaled, X_val_scaled, feature_scaler = self.scale_features(X_train=X_train, X_test=X_test, X_val=X_val)
        y_train_scaled, y_test_scaled, y_val_scaled, target_scaler = self.scale_targets(y_train=y_train, y_test=y_test, y_val=y_val)

        lstm_model, lstm_loss = self.LSTM_Model(
            X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, X_val_scaled=X_val_scaled,
            y_train_scaled=y_train_scaled, y_test_scaled=y_test_scaled, y_val_scaled=y_val_scaled)
        
        predictions = self.predict_future_values(X=X, model=lstm_model, output_steps=self.output_steps, feature_scaler=feature_scaler, target_scaler=target_scaler)
        timestamped_predictions = self.add_time_column_to_predicted_values(predictions=predictions, interval_minute=self.interval_minute)
        breakdown_probability = self.calculate_breakdown_probability(predictions=timestamped_predictions, column=self.target_column)
        results = self.calculate_model_performance(model=lstm_model, X_test_scaled=X_test_scaled, y_test=y_test, target_scaler=target_scaler)
        results['lstm_loss'] = lstm_loss
        results['breakdown_probability'] = breakdown_probability
        results['timestamp'] = datetime.now().replace(second=0, microsecond=0)
        return results, timestamped_predictions


    def preprocess(self, df):
        df.index = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%SZ')
        df.drop(inplace=True, axis=1, columns=['time', 'machine'])
        
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
        lstm_model.add(LSTM(100, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
        lstm_model.add(Dropout(0.3))
        lstm_model.add(LSTM(50))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(8, 'relu'))
        lstm_model.add(Dense(1, 'linear'))
        lstm_model.summary()

        # @TODO:patience argument will be updated as 10
        early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True) 
        checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
        lstm_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
        lstm_model.fit(X_train_scaled, y_train_scaled, validation_data=(X_val_scaled, y_val_scaled), epochs=2, batch_size=32, callbacks=[checkpoint, early_stopping])
        lstm_loss = lstm_model.evaluate(X_test_scaled, y_test_scaled)
        return lstm_model, lstm_loss
    

    def predict_future_values(self, X, model, output_steps:int, feature_scaler, target_scaler):
        feature_scaler = StandardScaler()
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
        
        metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R²": r2}
        self.logger.info(msg=f'metrics:\n{metrics}')
        return metrics


if __name__ == '__main__':
    df_15m = pd.read_csv('dataset/1724929644/dataset-90d.csv')[4::15]
    model = RNNModel()
    model.main(df=df_15m, input_days=90, output_days=10, interval_minute=15)