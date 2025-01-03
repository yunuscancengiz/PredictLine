# PredictLine

**Predictive Maintenance with Sensor Data**

This repository contains a predictive maintenance pipeline built to process sensor data from a production line. The project predicts potential machine breakdowns using advanced machine learning techniques and visualizes the results for decision-making.

---

## **Project Architecture**

The architecture is modular, incorporating several tools and technologies to ensure data flow and processing is efficient and reliable:

![Project Architecture](Flowcharts/Final%20Dark%20Flow%20Chart.jpg)

---

## **Grafana Dashboard Example**

The final results are visualized with Grafana as follows.

![Grafana Dashboard Example](Flowcharts/dashboard_example.PNG)

---

- **Data Sources:** Sensor data from InfluxDB.
- **Data Preprocessing:** Data cleaning and transformation in Apache Druid.
- **Kafka Topics:** Raw, processed, and predicted data managed through Apache Kafka.
- **Machine Learning:** LSTM-based deep learning model for time series prediction.
- **Databases:** Processed data in InfluxDB, predictions and metrics stored in PostgreSQL.
- **Visualization:** Real-time monitoring and insights via Grafana dashboards.

---

## **Getting Started**

Follow these steps to set up and run the project on your local or cloud server:

### **1. Prerequisites**
Ensure the following are installed:
- Docker
- Docker Compose

Install Docker:
```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run hello-world
```

---

### **2. Clone the Repository**
```bash
git clone https://github.com/yunuscancengiz/PredictLine.git
cd PredictLine
```

---

### **3. Set Up Environment Variables**
Create a `.env` file to store environment-specific configurations:
```bash
nano .env
```
Populate the `.env` file with the necessary configuration values as required by the project.
Necessary values:

```bash
# Server IP
GCP_IP=

# Corp. Influx Configuration
INFLUX_BUCKET=
INFLUX_ORG=
INFLUX_TOKEN=
INFLUX_URL=

# Personal Influx Configuration
MY_INFLUX_BUCKET=
MY_INFLUX_ORG=
MY_INFLUX_TOKEN=
MY_INFLUX_URL=

# PostgreSQL Configuration
POSTGRE_USERNAME=
POSTGRE_PASSWORD=
POSTGRE_HOST=
POSTGRE_PORT=
POSTGRE_DB_NAME=
```

---

### **4. Prepare Kafka Volume Path**
Create the Kafka data directory and assign appropriate permissions:
```bash
sudo mkdir -p ./kafka/data/kafka
sudo chmod -R 777 ./kafka/data/kafka
```

---

### **5. Start Services**
Bring up all services using Docker Compose:
```bash
sudo docker compose up -d
```

---

### **6. Create Kafka Topics**
Run the script to create Kafka topics:
```bash
sudo chmod +x create_kafka_topics.sh
./create_kafka_topics.sh
```

---

### **7. Send Initial Data to Kafka Topics**
Load the initial data into Kafka topics:
```bash
sudo chmod +x send_initial_data.sh
./send_initial_data.sh
```

---

### **8. Introduce Kafka Topics to Druid**
Configure Apache Druid to consume data from the Kafka topics using the Druid UI.

---

### **9. Build and Run the Main Application**
Build the Docker image for the main application and run it:
```bash
sudo docker build -t predictive-maintenance-app .
sudo docker run -d --rm --name app --network predictline_pm_pipeline_network predictive-maintenance-app
```

---

## **Features**

- **Data Pipeline:** Automated data ingestion, preprocessing, and visualization.
- **Machine Learning:** LSTM model for short-term (2 days) and long-term (10 days) predictions.
- **Retention Policies:** Efficient storage management in InfluxDB and Apache Druid.
- **Visualization:** Grafana dashboards for real-time insights.

---

## **Contributing**
Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contact**
For questions or collaboration opportunities, reach out to [Yunus Can Cengiz](https://github.com/yunuscancengiz).
