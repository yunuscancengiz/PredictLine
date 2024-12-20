#!/bin/bash

topics=(
  "raw-data"
  "raw-data-15m"
  "processed-data"
  "processed-data-15m"
  "predicted-data"
  "predicted-data-15m"
)

container_name="kafka"
kafka_topics_script="/opt/kafka/bin/kafka-topics.sh"
bootstrap_server="localhost:9092"

for topic in "${topics[@]}"; do
  echo "Creating topic: $topic"
  sudo docker exec -it $container_name $kafka_topics_script --create \
    --topic $topic \
    --bootstrap-server $bootstrap_server \
    --partitions 1 \
    --replication-factor 1

  if [ $? -eq 0 ]; then
    echo "Successfully created topic: $topic"
  else
    echo "Failed to create topic: $topic"
  fi
done
