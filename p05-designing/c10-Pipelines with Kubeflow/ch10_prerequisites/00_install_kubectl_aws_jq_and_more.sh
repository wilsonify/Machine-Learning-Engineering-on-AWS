#!/bin/bash

if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "Scripts needs to be run with sudo"
    exit
fi

set -o xtrace


./01_install_kubectl.sh

./02_install_aws_cli_v2.sh

./03_install_jq_and_more.sh

sudo -u $SUDO_USER ./04_check_prerequisites.sh

sudo -u $SUDO_USER ./05_additional_setup_instructions.sh

sudo -u $SUDO_USER ./06_remove_credentials_and_configure_variables.sh

./06_download_eksctl.sh

./07_install_kustomize.sh

