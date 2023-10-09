#!/bin/bash

sudo -u $SUDO_USER curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

sudo -u $SUDO_USER unzip awscliv2.zip

./aws/install

sudo -u $SUDO_USER aws --version

sudo -u $SUDO_USER rm -rf aws

sudo -u $SUDO_USER rm awscliv2.zip