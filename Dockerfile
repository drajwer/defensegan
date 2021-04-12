FROM tensorflow/tensorflow:1.7.0-gpu
ARG COMMAND
WORKDIR /defensegan

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD COMMAND