#!/usr/bin/env bash
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=true

virtualenv ~/.venv
source ~/.venv/bin/activate
make install


#append it to bash so every shell launches with it 
echo 'source ~/.venv/bin/activate' >> ~/.bashrc
echo 'export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True' >> ~/.bashrc

