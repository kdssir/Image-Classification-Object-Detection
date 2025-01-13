#FROM ubuntu:18.04
#FROM nvidia/cuda:9.0-base
FROM nvidia/cuda:9.0-devel

RUN apt-get update && \
    apt-get clean && \
    apt-get install -y vim && \
    apt-get update && \
    apt-get install wget

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3
RUN python3 -m pip install --upgrade pip
RUN apt install -y libsm6 libxext6 libxrender-dev
COPY . /opt/training/
WORKDIR /opt/training/
RUN apt install -y python3.6-tk
RUN pip3 install -r requirement.txt
