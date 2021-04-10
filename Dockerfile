FROM tensorflow/tensorflow:1.7.0-gpu
WORKDIR /defensegan

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD train.sh