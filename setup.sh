#!/usr/bin/env bash

virtualenv ~/.venv
source ~/.venv/bin/activate
echo 'export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=true' >> ~/.bashrc
source ~/.bashrc
make install


#append it to bash so every shell launches with it 
echo 'source ~/.venv/bin/activate' >> ~/.bashrc
