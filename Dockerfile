FROM python:3.10.4

WORKDIR /usr/src/app

COPY _requirements.txt ./
RUN pip install --upgrade --no-cache-dir -r _requirements.txt

RUN pip list --format=freeze > reqs.txt

COPY . .

CMD [ "python", "./call_test.py" ]