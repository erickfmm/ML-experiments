#FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
FROM ubuntu:22.04

WORKDIR /usr/src/app

RUN apt-get update && apt-get upgrade -y
RUN apt install pip graphviz unzip -y

COPY requirements.txt ./
RUN pip install --upgrade --no-cache-dir -r requirements.txt


# ARG TF_ENABLE_ONEDNN_OPTS=0

# RUN python3 -m spacy download en_core_news_lg
RUN python3 -m spacy download es_core_news_lg
RUN python3 -m spacy download es_core_news_sm
RUN python3 -m spacy download en_core_web_sm

COPY . .


CMD [ "python3", "./call_test.py" ]