FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update &&\
    apt-get install -y sudo wget unzip vim nano python3 python3-pip python3-venv tzdata libgmp3-dev
RUN wget -O - https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install_prereqs.sh | bash &&\
    wget -O - https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install.sh | bash

COPY ./requirements.txt ./certified-neural-approximations/requirements.txt

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install -r certified-neural-approximations/requirements.txt

COPY . ./certified-neural-approximations
WORKDIR /certified-neural-approximations

RUN pip install -e .
RUN ./download_data_files.sh

ENTRYPOINT [ "bash" ]