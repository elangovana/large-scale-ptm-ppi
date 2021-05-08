# SageMaker PyTorch image
ARG device=gpu
ARG account_url=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training
ARG cuda=cu101
FROM ${account_url}:1.4.0-${device}-py36${cuda}-ubuntu16.04

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY ./src  /opt/ml/code
RUN pip install -r /opt/ml/code/requirements.txt

