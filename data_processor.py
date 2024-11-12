import pandas as pd
import traceback
from _logger import ProjectLogger


class DataPreprocessor:
    thresholds = {
        'axialAxisRmsVibration': 0.1,
        'radialAxisKurtosis': 3,
        'radialAxisPeakAcceleration': 0.05,
        'radialAxisRmsAcceleration': 0.01
    }
    logger = ProjectLogger(class_name='DataPreprocessor').create_logger()

    def __init__(self):
        self.df = None


    def main(self, df:pd.DataFrame):
        try:
            self.df = df
            self.process()
            self.logger.info('Data pre-processed successfuly!')
            return self.df
        except Exception as e:
            self.logger.error(msg='Exception happened while pre-processing the data!')
            self.logger.error(msg=traceback.format_exc())

    
    def process(self):
        self.df['is_running'] = 1
        self.df.loc[
            (self.df['axialAxisRmsVibration'] < self.thresholds['axialAxisRmsVibration']) & 
            (self.df['radialAxisKurtosis'] < self.thresholds['radialAxisKurtosis']) & 
            (self.df['radialAxisPeakAcceleration'] < self.thresholds['radialAxisPeakAcceleration']) & 
            (self.df['radialAxisRmsAcceleration'] < self.thresholds['radialAxisRmsAcceleration']),
            'is_running'
        ] = 0
        return self.df

