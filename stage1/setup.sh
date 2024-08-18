#!/bin/bash
# set up a virtual environment to avoid library conflict

# necessary python libraries
pip install -r requirements.txt

sudo apt install python3-tk

python setup.py build develop

rm -rf build