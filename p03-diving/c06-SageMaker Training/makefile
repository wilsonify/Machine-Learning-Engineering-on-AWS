
all: clean deploy

clean:
	rm -rf tmp && mkdir -p tmp
	time aws s3 rm s3://s3_bucket --recursive

download:
	wget -O tmp/batch1.zip https://bit.ly/37zmQeb

unzip: download
	cd tmp && time unzip batch1.zip && time rm batch1.zip

upload: unzip
	time aws s3 cp --recursive tmp/. s3://{s3_bucket}/{prefix}/

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
