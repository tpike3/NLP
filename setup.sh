#!/bin/bash

# set up virtual environment
python3 -m venv env
echo 'Environment env created'
source env/bin/activate
echo 'env activated. Installing packages...'
pip install -r requirements.txt