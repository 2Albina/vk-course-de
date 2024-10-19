#!/bin/bash

# Запуск Spark-прилодения с мастером YARN и двумя экзекьюторами
spark-submit --master yarn --deploy-mode client --num-executors 2 << EOF

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import TfidfVectorizer
from pyspark.ml.regression import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import math

# Задача 1: Создаем Spark-сессию
spark = SparkSession.builder \
    .appName("Spark Experiments") \
    .config("spark.yarn.executor.memory", "2g") \
    .getOrCreate()

# Задача 2: Создание пустого файла на HDFS
spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(spark.sparkContext._jsc.hadoopConfiguration()).create(
    spark.sparkContext._jvm.org.apache.hadoop.fs.Path("/sparkExperiments.txt")
)

# Задача 3: Чтение таблиц ratings и tags, подсчет строк
ratings = spark.read.csv("/ml-latest-small/ratings.csv", header=True, inferSchema=True)
tags = spark.read.csv("/ml-latest-small/tags.csv", header=True, inferSchema=True)

ratings_count = ratings.count()
tags_count = tags.count()

stages = spark.sparkContext.statusTracker.getStageIds().length
tasks = spark.sparkContext.statusTracker.getTaskIds().length

with open("/sparkExperiments.txt", "a") as f:
    f.write(f"stages:{stages} tasks:{tasks}\n")

# Задача 4: Количество уникальных фильмов и пользователей
films_unique = ratings.select("movie).distinct().count()
users_unique = ratings.select("user").distinct().count()

with open("/sparkExperiments.txt", "a") as f:
    f.write(f"filmsUnique:{films_unique} usersUnique:{users_unique}\n")

# Задача 5: Подсчет оценок >= 4.0
good_ratings = ratings.filter(ratings["rating"] >= 4.0).count()

with open("/sparkExperiments.txt", "a") as f:
    f.write(f"goodRating:{good_ratings}\n")

# Задача 6: Средняя разница во времени между тегами и оценками
time_difference = tags.join(ratings, ["movie", "user"]).withColumn(
    "time_diff", (col("timestamp_x") - col("timestamp_y")).cast("double")
).select("time_diff").groupBy().avg().collect()[0][0]

with open("/sparkExperiments.txt", "a") as f:
    f.write(f"timeDifference:{time_difference}\n")

# Задача 7: Средняя оценка пользователей
avg_ratings = ratings.groupBy("userId").avg("rating").select("avg(rating)").groupBy().avg("avg(rating)").collect()[0][0]

with open("/sparkExperiments.txt", "a") as f:
    f.write(f"avgRating:{avg_ratings}\n")

# Задача 8: Обучение модели для предсказания оценок по тегам
tags_ratings = tags.join(ratings, ["movie", "user"]).select("tag", "rating")

# Преобразуем тег в числовое представление через TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tags_ratings.select("tag").rdd.flatMap(lambda x: x).collect())
y = tags_ratings.select("rating").rdd.flatMap(lambda x: x).collect()

# Обучаем модель
model = SGDRegressor()
model.fit(X, y)

# UDF для предсказания
def predict_rating(tag):
    vec = vectorizer.transform([tag])
    return float(model.predict(vec)[0])

predict_udf = udf(predict_rating, DoubleType())
predictions = tags_ratings.withColumn("predicted_rating", predict_udf(col("tag")))

# Вычисляем RMSE
rmse = math.sqrt(mean_squared_error(y, predictions.select("predicted_rating").rdd.flatMap(lambda x: x).collect()))

with open("/sparkExperiments.txt", "a") as f:
    f.write(f"rmse:{rmse}\n")

# Завершаем сессию
spark.stop()
EOF
