#!/bin/bash

sudo -u $SUDO_USER curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(sudo -u $SUDO_USER uname -s)_amd64.tar.gz" | sudo -u $SUDO_USER tar xz -C /tmp
mv -v /tmp/eksctl /usr/local/bin
sudo -u $SUDO_USER eksctl version