FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update &&\
    apt-get install -y sudo wget vim nano python3 python3-pip tzdata libgmp3-dev
RUN wget -O - https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install_prereqs.sh | bash &&\
    wget -O - https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install.sh | bash

COPY ./requirements.txt ./neural-abstraction/requirements.txt

RUN pip3 install -r neural-abstraction/requirements.txt

COPY . ./neural-abstraction
WORKDIR /neural-abstraction

ENTRYPOINT [ "bash" ]