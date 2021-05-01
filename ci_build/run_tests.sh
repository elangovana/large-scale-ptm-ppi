#!/usr/bin/env bash
 set -e

 echo "$@"

 VIRTUAL_ENV=$1
 TEST_REPORT_PATH=$2

 # Set up virtual env
 virtualenv -p python3 $VIRTUAL_ENV
 . $VIRTUAL_ENV/bin/activate

 #Install requirements
 pip install -r tests/requirements.txt
 pip install pyflakes==2.3.0

 #Run tests
 export PYTHONPATH=./src
 pytest tests --tb=short --junitxml=$TEST_REPORT_PATH

 #Run pyflakes to detect any import / syntax issues
 pyflakes ./**/*.py

 # Deactivate virtual envs
 deactivate