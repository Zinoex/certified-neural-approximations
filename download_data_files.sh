#! /bin/bash

filename="data/compression_ground_truth.zip"
fileid="1yFb23Qp4RFNAgg-WFW7Iq-sHdoyOeDhn"
wget -O ${filename} "https://drive.usercontent.google.com/download?id=${fileid}&export=download&confirm"

unzip -o ${filename} -d data/