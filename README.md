# PredictionFromSensorData

## Overview
This repository is a comprehensive data engineering project aimed at predictive maintenance using sensor data from industrial production lines. It handles data ingestion, processing, and visualization, leveraging tools like Apache Kafka, Apache Druid, PostgreSQL, and Python-based machine learning.

## Features
- **Data Collection**: Retrieves sensor data from InfluxDB and streams it using Kafka.
- **Data Processing**: Preprocesses data using Apache Druid and Python scripts.
- **Machine Learning**: Trains predictive models to anticipate machine breakdowns.
- **Data Storage**: Stores processed data and model results in PostgreSQL.
- **Visualization**: Integrates with Grafana for real-time monitoring and insights.

## Project Structure
- **_create_dataset.py**: Script to generate synthetic datasets for model training.
- **_initial_data.py**: Initializes and loads starting data into the pipeline.
- **_logger.py**: Handles logging across scripts.
- **consumer.py**: Consumes data from Kafka topics.
- **data_processor.py**: Preprocesses data for machine learning.
- **druid_cleaner.py**: Cleans up older data in Druid for optimal performance.
- **main.py**: The main execution script for end-to-end data flow.
- **model.py**: Defines and trains machine learning models.
- **postgre_db.py**: Interfaces with PostgreSQL for data storage.
- **producer.py**: Streams data to Kafka topics.

## Prerequisites
- **Python 3.8+**
- **Docker**
- **Apache Kafka**
- **Apache Druid**
- **PostgreSQL**

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yunuscancengiz/PredictionFromSensorData.git
   cd PredictionFromSensorData
