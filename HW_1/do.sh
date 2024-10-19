#!/bin/bash

hdfs dfs -mkdir /createme
hdfs dfs -rm -r /delme
echo "Some content" | hdfs dfs -put - /nonnull.txt
hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar wordcount /shadow.txt /output
hdfs dfs -cat /output/* | grep -w "Innsmouth" | awk '{print $2}' | hdfs dfs -put - /whataboutinsmouth.txt
