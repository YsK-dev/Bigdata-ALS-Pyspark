# PySpark ALS Recommendation System

This repository contains a PySpark-based recommendation system using Alternating Least Squares (ALS) for collaborative filtering. The goal is to predict user ratings for movies and provide recommendations based on those predictions. The dataset used in this project is the MovieLens dataset, which contains movie ratings from users.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Code Explanation](#code-explanation)
4. [Deep Dive into ALS](#deep-dive-into-als)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction

Recommendation systems are widely used in various applications like e-commerce, streaming services, and social media platforms. Collaborative filtering is one of the most popular techniques used in recommendation systems. ALS (Alternating Least Squares) is a matrix factorization algorithm that is particularly well-suited for collaborative filtering tasks.

This project demonstrates how to implement an ALS-based recommendation system using PySpark. The code includes data loading, preprocessing, model training, evaluation, and making recommendations.

## Dataset

The dataset used in this project is the MovieLens dataset, which includes:

- **ratings.csv**: Contains user ratings for movies.
- **movies.csv**: Contains movie titles and genres.

### Dataset Schema

- **ratings.csv**:
  - `userId`: Unique identifier for the user.
  - `movieId`: Unique identifier for the movie.
  - `rating`: Rating given by the user to the movie.
  - `timestamp`: Timestamp when the rating was given.

- **movies.csv**:
  - `movieId`: Unique identifier for the movie.
  - `title`: Title of the movie.
  - `genres`: Genres associated with the movie.

## Code Explanation

### 1. Setting Up PySpark

The code starts by setting up a PySpark session. PySpark is the Python API for Apache Spark, which is a distributed computing framework. The session is configured with specific memory settings to handle large datasets efficiently.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName('Recommender_system') \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "1024m") \
    .getOrCreate()
```

### 2. Loading the Dataset

The dataset is loaded into PySpark DataFrames. The `ratings.csv` and `movies.csv` files are read and their schemas are inferred.

```python
ratings = spark.read.csv("/Users/ysk/Downloads/archive(3)/ratings.csv", header=True, inferSchema=True)
movies = spark.read.csv("/Users/ysk/Downloads/archive(3)/movies.csv", header=True, inferSchema=True)
```

### 3. Data Exploration

Basic data exploration is performed to understand the dataset. This includes checking the schema, counting the number of rows, and displaying the first few rows of the dataset.

```python
ratings.printSchema()
movies.printSchema()
print("Ratings dataset shape: Rows =", ratings.count(), ", Columns =", len(ratings.columns))
print("Movies dataset shape: Rows =", movies.count(), ", Columns =", len(movies.columns))
```

### 4. Data Preprocessing

The dataset is split into training and testing sets. The training set is used to train the ALS model, and the testing set is used to evaluate the model's performance.

```python
train_data, test_data = ratings.randomSplit([0.7, 0.3], seed=5033)
```

### 5. Model Training

The ALS model is trained using the training dataset. The model is configured with parameters such as the number of iterations, regularization parameter, and columns for user, item, and rating.

```python
from pyspark.ml.recommendation import ALS

als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(train_data)
```

### 6. Model Evaluation

The model's performance is evaluated using the Root Mean Squared Error (RMSE) metric. The predictions are made on the test dataset, and the RMSE is calculated.

```python
from pyspark.ml.evaluation import RegressionEvaluator

predictions = model.transform(test_data)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f'Root Mean Squared Error (RMSE): {rmse}')
```

### 7. Making Recommendations

The trained model is used to make recommendations for a specific user. The recommendations are sorted by the predicted rating.

```python
user_recommendations = model.recommendForAllUsers(10)
user_recommendations.show()
```

## Deep Dive into ALS

### What is ALS?

Alternating Least Squares (ALS) is a matrix factorization algorithm used in collaborative filtering. It decomposes the user-item interaction matrix into two lower-dimensional matrices: one representing users and the other representing items. The goal is to minimize the difference between the observed ratings and the predicted ratings.

### How ALS Works

1. **Matrix Factorization**: The user-item interaction matrix \( R \) is factorized into two matrices \( U \) (user factors) and \( V \) (item factors) such that \( R \approx U \times V^T \).

2. **Alternating Optimization**: The algorithm alternates between fixing \( U \) and optimizing \( V \), and fixing \( V \) and optimizing \( U \). This process continues until convergence.

3. **Regularization**: To prevent overfitting, a regularization term is added to the loss function.

### Advantages of ALS

- **Scalability**: ALS is highly scalable and can handle large datasets efficiently.
- **Parallelization**: The algorithm can be parallelized, making it suitable for distributed computing frameworks like Spark.
- **Implicit Feedback**: ALS can handle both explicit (ratings) and implicit (clicks, views) feedback.

### Challenges

- **Cold Start Problem**: ALS struggles with new users or items that have no interaction history.
- **Hyperparameter Tuning**: The performance of ALS depends on the choice of hyperparameters like the number of factors and regularization parameter.

## Results

The model's performance is evaluated using the RMSE metric. A lower RMSE indicates better performance. The recommendations generated by the model can be used to suggest movies to users based on their predicted ratings.

## Conclusion

This project demonstrates how to build a recommendation system using ALS in PySpark. The model is trained on the MovieLens dataset and evaluated using RMSE. The recommendations generated by the model can be used to enhance user experience by suggesting relevant movies.

## References

- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Collaborative Filtering with ALS](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)

---

This README provides a comprehensive overview of the code and the ALS algorithm. It explains the steps involved in building a recommendation system and provides insights into the underlying principles of collaborative filtering using ALS.
