#!/bin/bash

docker build -t drejerk/defensegan .
docker run -d -v "$(pwd)":/defensegan -it --gpus all --runtime=nvidia --name defensegan-debug-bdpa drejerk/defensegan /bin/bash -c "./debug-long.sh"
docker logs -f defensegan-debug-bdpa

docker run -d -v "$(pwd)":/defensegan -it --gpus all --runtime=nvidia --name defensegan-debug-bdpa-2 drejerk/defensegan /bin/bash -c "./debug-long.sh"
docker logs -f defensegan-debug-bdpa-2
