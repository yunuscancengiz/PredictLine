import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class RNNModel:
    def __init__(self, filename:str, input_steps:int=500, output_steps:int=50):
        self.df = pd.read_csv(filename)
        self.processed_df = self.preprocess(df=self.df)
        self.input_steps = input_steps
        self.output_steps = output_steps

        # normalize data
        self.input_columns = ['axialAxisRmsVibration', 'radialAxisKurtosis', 'radialAxisPeakAcceleration', 'radialAxisRmsAcceleration', 'radialAxisRmsVibration', 'temperature', 'is_running']
        self.scaler = MinMaxScaler()
        self.scaled_df = self.scaler.fit_transform(self.processed_df[self.input_columns])

        # Find the index of the target column
        self.target_index = self.input_columns.index('axialAxisRmsVibration')


    def main(self):
        X, y = self.prepare_data(df=self.scaled_df)
        X_train, X_test, y_train, y_test = self.split_and_reshape(X=X, y=y)
        lstm_loss = self.LSTM_Model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        print(lstm_loss)


    def preprocess(self, df):
        thresholds = {
            'axialAxisRmsVibration': 0.1,
            'radialAxisKurtosis': 3,
            'radialAxisPeakAcceleration': 0.05,
            'radialAxisRmsAcceleration': 0.01
        }
        df['is_running'] = 1
        df.loc[
            (df['axialAxisRmsVibration'] < thresholds['axialAxisRmsVibration']) & 
            (df['radialAxisKurtosis'] < thresholds['radialAxisKurtosis']) & 
            (df['radialAxisPeakAcceleration'] < thresholds['radialAxisPeakAcceleration']) & 
            (df['radialAxisRmsAcceleration'] < thresholds['radialAxisRmsAcceleration']),
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
        # Split the data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Reshape input data to 3D for LSTM/GRU [samples, timesteps, features]
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

        # Evaluate models on test data
        lstm_loss = lstm_model.evaluate(X_test, y_test)
        return lstm_loss
    

if __name__ == '__main__':
    model = RNNModel(filename='dataset/1724929644/dataset-60d-30d.csv')
    model.main()