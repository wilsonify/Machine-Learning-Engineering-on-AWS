#!/bin/bash

apt install jq gettext bash-completion moreutils -y

sudo -u $SUDO_USER echo 'yq() {
   docker run --rm -i -v "${PWD}":/workdir mikefarah/yq "$@"
}' | tee -a ~/.bashrc && source ~/.bashrc