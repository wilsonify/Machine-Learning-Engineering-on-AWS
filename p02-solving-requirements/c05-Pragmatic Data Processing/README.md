

Pragmatic Data Processing and Analysis
====

This chapter focuses on the different services available 
such as AWS Glue DataBrew and Amazon SageMaker Data Wrangler 
when working on data processing and analysis requirements.

### I. Links

### II. Commands

#### Automating Data Preparation and Analysis with AWS Glue DataBrew

##### Verifying the Results

```
TARGET=<PASTE COPIED S3 URL>
aws s3 cp $TARGET bookings.csv
head bookings.csv
```

#### ➤ Preparing ML Data with Amazon SageMaker Data Wrangler

##### Transforming the Data

```
df = df.filter(df.children >= 0)
expression = df.booking_changes > 0
df = df.withColumn('has_booking_changes', expression)
```

##### Verifying the Results

```
mv * /tmp 2>/dev/null
S3_PATH=<PASTE S3 URL>
aws s3 cp $S3_PATH/ . --recursive
ls -R
head */default/*.csv
```
