FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y

RUN apt install -y python3-pip python3-dev
RUN python3 -m pip install --upgrade pip

COPY requirement.txt /opt/requirements.txt

WORKDIR /opt/

RUN pip3 install -r requirements.txt

RUN apt install -y libsm6 libxext6 libxrender-dev

