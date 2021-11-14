#!/bin/bash
pip install -r requirements.txt

dateNow=$(date '+%Y%m%d%H%M')  
path=logs/$dateNow
mkdir $path

./evaluate-long.sh 1> $path/output.log 2> $path/error.log
