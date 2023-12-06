#!/usr/bin/env bash

virtualenv ~/.venv
source ~/.venv/bin/activate
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
make install


#append it to bash so every shell launches with it 
echo 'source ~/.venv/bin/activate' >> ~/.bashrc
echo 'export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True' >> ~/.bashrc

