#!/bin/bash

docker build -t drejerk/defensegan .
docker run -v "$(pwd)":/defensegan -d --name defensegan drejerk/defensegan
docker logs -f defensegan