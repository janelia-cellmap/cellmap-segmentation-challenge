FROM python:3.11

COPY . /opt/repo

RUN pip install --no-cache /opt/repo