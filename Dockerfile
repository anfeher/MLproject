FROM python:3.12.1-slim-bullseye

WORKDIR /app

RUN apt update -y

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY artifacts artifacts
COPY src src
COPY templates templates
COPY app.py app.py

CMD ["python3","app.py"]