#PYSPARK CODE: 

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, min, max, mean, translate
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import avg
from pyspark.sql.functions import sum as pyspark_sum
from pyspark.sql.functions import to_date
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import pandas as pd

# Create a SparkSession
spark = SparkSession.builder.appName("NullValues").getOrCreate()

# Read the dataset into a DataFrame from HDFS Dir location
file_path = "hdfs:///user/chhabraridhima118/data/Data.csv"
df = spark.read.format("csv").option("header", "true").load(file_path)

#DATA EXPLORATION :
# to check the top values in the DF:
df.head()

#Describe the DF: 
df.describe()

#structure of the dataframe: 
df.printSchema()

# to SELECT A PARTICULAR COLUMN:
from pyspark.sql.functions import col
selected_column = df.select(col('Year'))
selected_column.show()

#DATA CLEANING:
#COUNT THE NULL VALUES in the Dataset:
null_counts = df.select([pyspark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])

# Display the counts
null_counts.show()

# remove null values:
df = df.dropna(subset=['Value', 'Lower', 'Upper'])
df = df.drop('UNICEF Region')

# TO Count <100 , <200, <500 String values from each dataset
desired_strings = ['<200', '<100', '<500','<0.01']
filtered_df = df.filter(col('Value').isin(desired_strings))
filtered_df.count()

#DATA TRANSFORMATION:
# converting the string values to Numeric values:

string_mapping = {'<100': 50,'<200':100,'<500':250,'<0.01':0}

df = df.withColumn(
    'Value',
    when(df['Value'] == '<100', string_mapping['<100'])
    .when(df['Value'] == '<200', string_mapping['<200'])
    .when(df['Value'] == '<500', string_mapping['<500'])
    .when(df['Value'] == '<0.01', string_mapping['<0.01'])
    .otherwise(df['Value'])  # If none of the conditions match, keep the original value
)
df = df.withColumn(
    'Lower',
    when(df['Lower'] == '<100', string_mapping['<100'])
    .when(df['Lower'] == '<200', string_mapping['<200'])
    .when(df['Lower'] == '<500', string_mapping['<500'])
    .when(df['Lower'] == '<0.01', string_mapping['<0.01'])
    .otherwise(df['Lower'])  # If none of the conditions match, keep the original value
)
df = df.withColumn(
    'Upper',
    when(df['Upper'] == '<100', string_mapping['<100'])
    .when(df['Upper'] == '<200', string_mapping['<200'])
    .when(df['Upper'] == '<500', string_mapping['<500'])
    .when(df['Upper'] == '<0.01', string_mapping['<0.01'])
    .otherwise(df['Upper'])  # If none of the conditions match, keep the original value
)

# Assuming 'df' is your DataFrame and 'string_column' is the column with string values like '37,50,000'
df = df.withColumn('Value', translate(col('Value'), ',', '').cast('int'))
df = df.withColumn('Upper', translate(col('Upper'), ',', '').cast('int'))
df = df.withColumn('Lower', translate(col('Lower'), ',', '').cast('int'))

#--------------------------------------------------------------------------------------------------------------------#
# Analysis 
#Mean of all the indicators:
# create separate dataframes for different indicators:
IncidenceRate = df.filter(df['Indicator'] == 'Estimated incidence rate (new HIV infection per 1,000 uninfected population)')
AnnualDeaths = df.filter(df['Indicator'] == 'Estimated number of annual AIDS-related deaths')
RateOfAnnualDeaths = df.filter(df['Indicator'] == 'Estimated rate of annual AIDS-related deaths (per 100,000 population)')
AnnualNewInfection = df.filter(df['Indicator'] == 'Estimated number of annual new HIV infections')
WithInfection = df.filter(df['Indicator'] == 'Estimated number of people living with HIV')
TransmissionRate = df.filter(df['Indicator'] == 'Estimated mother-to-child transmission rate (%)')


# full dataset Mean Min Max:
#Value Column
min_value = df.select(min('Value')).collect()[0][0]
max_value = df.select(max('Value')).collect()[0][0]
mean_value = df.select(mean('Value')).collect()[0][0]


#UPPER:
min_value_Upper = df.select(min('Upper')).collect()[0][0]
max_value_Upper = df.select(max('Upper')).collect()[0][0]
mean_value_Upper = df.select(mean('Upper')).collect()[0][0]


#LOWER:
min_value_Lower = df.select(min('Lower')).collect()[0][0]
max_value_Lower = df.select(max('Lower')).collect()[0][0]
mean_value_Lower = df.select(mean('Lower')).collect()[0][0]


#FOR ALL 6 INDICATORS:
#AnnualDeaths:
max_value = AnnualDeaths.select(max('Value')).collect()[0][0]
min_value = AnnualDeaths.select(min('Value')).collect()[0][0]
mean_value = AnnualDeaths.select(mean('Value')).collect()[0][0]

#IncidenceRate
max_value = IncidenceRate.select(max('Value')).collect()[0][0]
min_value = IncidenceRate.select(min('Value')).collect()[0][0]
mean_value = IncidenceRate.select(mean('Value')).collect()[0][0]

#RateOfAnnualDeaths
max_value = RateOfAnnualDeaths.select(max('Value')).collect()[0][0]
min_value = RateOfAnnualDeaths.select(min('Value')).collect()[0][0]
mean_value = RateOfAnnualDeaths.select(mean('Value')).collect()[0][0]

#AnnualNewInfection
max_value = AnnualNewInfection.select(max('Value')).collect()[0][0]
min_value = AnnualNewInfection.select(min('Value')).collect()[0][0]
mean_value = AnnualNewInfection.select(mean('Value')).collect()[0][0]

#WithInfection
max_value = WithInfection.select(max('Value')).collect()[0][0]
min_value = WithInfection.select(min('Value')).collect()[0][0]
mean_value = WithInfection.select(mean('Value')).collect()[0][0]

#TransmissionRate
max_value = TransmissionRate.select(max('Value')).collect()[0][0]
min_value = TransmissionRate.select(min('Value')).collect()[0][0]
mean_value = TransmissionRate.select(mean('Value')).collect()[0][0]

#Range of the dataset for Value column
range_values = df.select((max('Value') - min('Value')).alias('range')).collect()[0]['range']

#TREND ANALYSIS ON DATA:


spark = SparkSession.builder.appName("YearStringToTimestamp").getOrCreate()
Trend_df = df
Trend_df = Trend_df.withColumn('Year', to_date(col('Year'), 'yyyy'))
Trend_df
Trend_df.show()

windowSpec = Window.orderBy('timestamp').rowsBetween(-7, 0)
Trend_df = Trend_df.withColumn('Year', col('Year').cast('timestamp'))

Trend_df = Trend_df.withColumn('moving_avg', avg('Value').over(windowSpec))



Trend_df.select('Year', 'Value', 'moving_avg').show()


#TREND ANALYSIS: 
#trend of the 'Value' column over the years
spark = SparkSession.builder.appName("TrendAnalysis").getOrCreate()
trend_analysis = Trend_df.groupBy('Year').agg(avg('Value').alias('Average_Value'))
trend_analysis = trend_analysis.orderBy('Year')
trend_analysis.show()

# MACHINE LEARNING: 



# Create a SparkSession
spark = SparkSession.builder.appName("LinearRegression").getOrCreate()

# Define categorical columns for encoding
categorical_columns = ['Sex', 'Age']

# Indexing and encoding categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(df) for col in categorical_columns]
encoder = OneHotEncoder(inputCols=[indexer.getOutputCol() for indexer in indexers], outputCols=[col+"_encoded" for col in categorical_columns])
assembler = VectorAssembler(inputCols=encoder.getOutputCols() + ['Lower', 'Upper'], outputCol="features")

# Combine all stages into a pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler])

# Fit and transform the data
model = pipeline.fit(df)
df = model.transform(df)

# Cast 'Year' column to integer
df = df.withColumn("Year", df["Year"].cast("integer"))

# Split the data into training and testing sets
train_data, test_data = df.randomSplit([0.7, 0.3])  # 70% for training, 30% for testing

# Initialize Linear Regression model
lr = LinearRegression(featuresCol='features', labelCol='Value')

# Train the model
model = lr.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluation = model.evaluate(test_data)
print("RMSE:", evaluation.rootMeanSquaredError)
print("Root Mean Squared Error (RMSE):", evaluation.rootMeanSquaredError)
print("Mean Squared Error (MSE):", evaluation.meanSquaredError)
print("Mean Absolute Error (MAE):", evaluation.meanAbsoluteError)
print("R-squared (R2):", evaluation.r2)
predictions.select("Value", "prediction").show()


# Create a SparkSession
spark = SparkSession.builder.appName("LinearRegression_annualDeaths").getOrCreate()

# Define categorical columns for encoding
categorical_columns = ['Sex', 'Age']


# Indexing and encoding categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(AnnualDeaths) for col in categorical_columns]
encoder = OneHotEncoder(inputCols=[indexer.getOutputCol() for indexer in indexers], outputCols=[col+"_encoded" for col in categorical_columns])
assembler = VectorAssembler(inputCols=encoder.getOutputCols() + ['Year','Lower', 'Upper'], outputCol="features")

# Combine all stages into a pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler])

# Fit and transform the data
model = pipeline.fit(AnnualDeaths)
AnnualDeaths = model.transform(AnnualDeaths)

# Cast 'Year' column to integer
AnnualDeaths = AnnualDeaths.withColumn("Year", AnnualDeaths["Year"].cast("integer"))

# Split the data into training and testing sets
train_data, test_data = AnnualDeaths.randomSplit([0.7, 0.3])  # 70% for training, 30% for testing

# Initialize Linear Regression model
lr = LinearRegression(featuresCol='features', labelCol='Value')

# Train the model
model = lr.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluation = model.evaluate(test_data)
print("RMSE:", evaluation.rootMeanSquaredError)
print("Root Mean Squared Error (RMSE):", evaluation.rootMeanSquaredError)
print("Mean Squared Error (MSE):", evaluation.meanSquaredError)
print("Mean Absolute Error (MAE):", evaluation.meanAbsoluteError)
print("R-squared (R2):", evaluation.r2)
predictions.select("Value", "prediction").show()
