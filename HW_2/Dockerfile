# Используем базовый образ для Spark
FROM openjdk:8

# Установка Hadoop
RUN wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz && \
    tar -xzvf hadoop-3.3.6.tar.gz && \
    mv hadoop-3.3.6 /usr/local/hadoop

# Установка Spark
RUN wget https://dlcdn.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz  && \
    tar -xvf spark-3.5.3-bin-hadoop3.tgz && \
    mv spark-3.5.3-bin-hadoop3 /usr/local/spark

# Установка библиотек для машинного обучения
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install pyspark scikit-learn

# Установка переменных окружения
ENV HADOOP_HOME=/usr/local/hadoop
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# Копируем скрипт с командами
COPY do.sh /usr/local/spark/do.sh
RUN chmod +x /usr/local/spark/do.sh

# Открытие портов
EXPOSE 8088 8042 8020 50070 18080

# Запуск HDFS, YARN и Spark
CMD ["/bin/bash", "-c", "start-dfs.sh && start-yarn.sh && /usr/local/spark/do.sh"]
