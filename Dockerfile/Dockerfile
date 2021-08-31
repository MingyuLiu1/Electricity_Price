FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    python-qt4 \
    python-pyside \
    python-pip \
    python3-pip \
    python3-pyqt5

RUN apt install -y yum-utils
RUN apt-get install -y git
RUN apt-get install -y vim 

RUN git clone https://github.com/MingyuLiu1/Electricity_Price.git 
RUN cd /Users/apple/Desktop/Dockerfile
RUN python docker_test.py

RUN pip install --upgrade pip && pip install argparse matplotlib torch scikit-learn numpy scipy pandas requests pytorch_lightning pytorch_forecasting requests datetime

RUN echo "Seccessfully create docker container"
