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
CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait whitebox.py \
--cfg output/gans/mnist-long \ 
--results_dir debugging \
--bb_model A \
--sub_model B \
--fgsm_eps 0.3 \
--defense_type defense_gan \
--attack_type spsa \
--debug True \
--debug_dir debug/debugging-mnist-spsa \
--nb_epochs 1 \
--nb_epochs_s 1 \
--data_aug 1 \ 