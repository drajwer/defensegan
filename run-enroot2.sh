#!/bin/bash

sleep 6h;

dateNow=$(date '+%Y%m%d%H%M')  
path=logs/$dateNow-$1
mkdir $path

NVIDIA_VISIBLE_DEVICES=$2
CUDA_VISIBLE_DEVICES=$2
export NVIDIA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES
./evaluate-long-5.sh $1 1> $path/output.log 2> $path/error.log
