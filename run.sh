#!/bin/bash
container_name='defensegan-train-and-measure-2'

docker build -t drejerk/defensegan .
docker run -d -v "$(pwd)":/defensegan -it --gpus all --runtime=nvidia --name $container_name drejerk/defensegan /bin/bash -c "./train-and-measure.sh"
docker logs -f $container_name
