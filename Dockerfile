FROM pytorch/pytorch:latest

RUN apt-get update && \
    apt-get install -y \
    tmux \
    htop
    
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt
