#!/bin/bash

docker build -t drejerk/defensegan .
docker run -d -v "$(pwd)":/defensegan -it --gpus all --runtime=nvidia --name defensegan-debug-bdpa drejerk/defensegan /bin/bash -c "./evaluate-long.sh"
docker logs -f defensegan-eval-long
