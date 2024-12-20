#!/bin/bash

msg1='{"machine":"Blower-Pump-1", "time":"2020-01-01T00:00:00Z", "axialAxisRmsVibration":0.128, "radialAxisKurtosis":3.020, "radialAxisPeakAcceleration":0.024, "radialAxisRmsAcceleration":0.006, "radialAxisRmsVibration":0.126, "temperature":26.2}'
msg2='{"machine":"Blower-Pump-1", "time":"2020-01-01T00:00:00Z", "axialAxisRmsVibration":0.128, "radialAxisKurtosis":3.020, "radialAxisPeakAcceleration":0.024, "radialAxisRmsAcceleration":0.006, "radialAxisRmsVibration":0.126, "temperature":26.2, "is_running":1}'

broker="localhost:9092"

echo "$msg1" | sudo docker exec -i kafka /opt/kafka/bin/kafka-console-producer.sh --topic raw-data --broker-list $broker
echo "$msg1" | sudo docker exec -i kafka /opt/kafka/bin/kafka-console-producer.sh --topic raw-data-15m --broker-list $broker

echo "$msg2" | sudo docker exec -i kafka /opt/kafka/bin/kafka-console-producer.sh --topic processed-data --broker-list $broker
echo "$msg2" | sudo docker exec -i kafka /opt/kafka/bin/kafka-console-producer.sh --topic processed-data-15m --broker-list $broker

echo "Messages sent successfully!"
