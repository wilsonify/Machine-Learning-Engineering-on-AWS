BUCKET_NAME="<INSERT BUCKET NAME HERE>"
aws s3 cp tmp/bookings.train.csv s3://$BUCKET_NAME/datasets/bookings/bookings.train.csv
aws s3 cp tmp/bookings.test.csv s3://$BUCKET_NAME/datasets/bookings/bookings.test.csv
