# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow:1.7.0-gpu as base

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /defensegan
COPY . /defensegan


###########START NEW IMAGE : DEBUGGER ################### 
FROM base as debug
RUN pip install ptvsd

WORKDIR /defensegan/
# CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait whitebox.py \
# --cfg output/gans/mnist-long \ 
# --results_dir debugging \
# --model A \
# --fgsm_eps 0.3 \
# --defense_type defense_gan \
# --attack_type bpda-fgsm \
# --nb_epochs 10
#ENV CUDA_VISIBLE_DEVICES=-1
ENV set=mnist

CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait \
measure_gan.py --cfg output/gans/$set-long-measure --results_dir train_and_measure_gan_$set_debug --probe_size 500 --iter 999

#--debug True \
# --debug_dir debug/debugging-mnist-bpda1 \5