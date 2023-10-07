
clean:
	rm -rf tmp

tmp:
	mkdir -p tmp

tmp/knn.model.tar.gz:
	wget -O tmp/knn.model.tar.gz https://github.com/wilsonify/Machine-Learning-Engineering-on-AWS/raw/main/p04-securing/c08-Model%20Monitoring/knn.model.tar.gz

tmp/ll.model.tar.gz:
	wget -O tmp/ll.model.tar.gz https://github.com/wilsonify/Machine-Learning-Engineering-on-AWS/raw/main/p04-securing/c08-Model%20Monitoring/ll.model.tar.gz

tmp/baseline.csv:
	wget -O tmp/baseline.csv https://bit.ly/3td5vjx

upload: tmp/knn.model.tar.gz tmp/ll.model.tar.gz tmp/baseline.csv
	aws s3 cp tmp/knn.model.tar.gz s3://064592191516-ml-engineering/c08/knn.model.tar.gz
	aws s3 cp tmp/ll.model.tar.gz s3://064592191516-ml-engineering/c08/ll.model.tar.gz
	aws s3 cp tmp/baseline.csv s3://064592191516-ml-engineering/c08/baseline.csv
