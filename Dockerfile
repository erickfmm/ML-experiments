#FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
FROM ubuntu:22.04

WORKDIR /usr/src/app

RUN apt-get update && apt-get upgrade -y
RUN apt install graphviz unzip python3-tk -y

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt ./
RUN uv venv && uv pip install --no-cache -r requirements.txt

RUN uv run python3 -m spacy download es_core_news_lg
RUN uv run python3 -m spacy download es_core_news_sm
RUN uv run python3 -m spacy download en_core_web_sm

COPY . .

# Default: run the web GUI server
# Access at http://localhost:3004
EXPOSE 3004
CMD [ "uv", "run", "python3", "./run_web_gui.py", "--host", "0.0.0.0", "--port", "3004" ]