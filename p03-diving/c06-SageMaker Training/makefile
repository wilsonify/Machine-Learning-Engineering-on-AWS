
all: clean deploy

clean:
	rm -rf tmp && mkdir -p tmp
	rm -rf tmp2 && mkdir -p tmp2
	aws s3 rm s3://s3_bucket --recursive
	aws s3 rm s3://s3_bucket --recursive

download:
	wget -O tmp/batch1.zip https://github.com/wilsonify/Machine-Learning-Engineering-on-AWS/raw/main/p03-diving/c06-SageMaker%20Training/batch1.zip
	wget -O tmp2/batch2.zip https://github.com/wilsonify/Machine-Learning-Engineering-on-AWS/raw/main/p03-diving/c06-SageMaker%20Training/batch2.zip

unzip: download
	cd tmp && unzip batch1.zip && rm batch1.zip && cd ..
	cd tmp2 && unzip batch2.zip && rm batch2.zip && cd ..

upload: unzip
	aws s3 cp --recursive tmp/. s3://064592191516-ml-engineering/c06/
	aws s3 cp --recursive tmp2/. s3://064592191516-ml-engineering/c06/

split: upload
	python split.py

train: split
	python train.py

evaluate: train
	python evaluate.py

validate: train
	python validate.py

deploy: validate
	python deploy.py

plot_images: deploy
	python plot_images.py


login:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 064592191516.dkr.ecr.us-east-1.amazonaws.com