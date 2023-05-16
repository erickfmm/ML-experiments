FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get update && apt-get upgrade -y
RUN apt install pip graphviz -y

RUN pip install --upgrade --no-cache-dir -r requirements.txt

COPY . .

#RUN python3 -m spacy download es_core_news_lg

CMD [ "python3", "./call_test.py" ]