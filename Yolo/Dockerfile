FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget git && apt-get clean && rm -rf /var/cache/apt
RUN apt-get -y autoremove && apt-get -y autoclean
RUN rm -rf /var/cache/apt

RUN apt install -y python3-pip python3-dev
RUN python3 -m pip install --upgrade pip
RUN apt install -y libopencv-dev
RUN pip3 install numpy
RUN apt install -y vim

RUN apt-get update
RUN cd /opt
WORKDIR /opt
RUN git clone https://github.com/AlexeyAB/darknet.git
WORKDIR /opt/darknet
RUN sed -i s/GPU=0/GPU=1/g Makefile
RUN sed -i s/OPENCV=0/OPENCV=1/g Makefile
RUN sed -i s/CUDNN_HALF=0/CUDNN_HALF=1/g Makefile
RUN sed -i s/CUDNN=0/CUDNN=1/g Makefile
RUN sed -i s/LIBSO=0/LIBSO=1/g Makefile
RUN make

RUN mkdir -p VOCdevkit/VOC2007

RUN cd ../
RUN mkdir training_scripts exported_model logs
RUN cd training_scripts

COPY src/config_overhaul.py /opt/training_scripts/config_overhaul.py
COPY src/yolo_trainer.sh /opt/training_scripts/yolo_trainer.sh
COPY src/label_prompting.py /opt/training_scripts/label_prompting.py
COPY src/chop_utility.py /opt/training_scripts/chop_utility.py
COPY src/anchor_creation.py  /opt/training_scripts/anchor_creation.py
