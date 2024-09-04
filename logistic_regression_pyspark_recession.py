from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize the Spark Session
spark = SparkSession.builder.appName("RecessionPrediction").getOrCreate()

# Load the Data
df = spark.read.csv("archive/US_Recession.csv", header=True, inferSchema=True)
df = df.drop('Unnamed: 0')  # Drop unnecessary column

# Handle Missing Values
df = df.dropna()

# Separate Predictors and Target Variable
assembler = VectorAssembler(inputCols=[col for col in df.columns if col != 'Recession'], outputCol='features')
df = assembler.transform(df).select('features', 'Recession')

# Feature Scaling
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Train-Test Split
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Initialize the Logistic Regression Model
lr = LogisticRegression(featuresCol='scaled_features', labelCol='Recession')

# Hyperparameter Tuning
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol='Recession'),
                          numFolds=5)

# Model Training
cv_model = crossval.fit(train_df)

# Model Evaluation
predictions = cv_model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol='Recession')
auc = evaluator.evaluate(predictions)
print(f"Model AUC: {auc:.2f}")

# Feature Importance
best_model = cv_model.bestModel
coefficients = best_model.coefficients
print("Feature Importances:", coefficients)
