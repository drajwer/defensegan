#!/bin/bash

docker build -t drejerk/defensegan .
docker run -d -v "$(pwd)":/defensegan -it --cpus="4.0" --gpus 4 --runtime=nvidia --name defensegan-eval-bpda drejerk/defensegan /bin/bash -c "./evaluate-long-2.sh"
docker logs -f defensegan-eval-bpda