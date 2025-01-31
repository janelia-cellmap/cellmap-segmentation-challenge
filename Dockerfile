FROM python:3.12

COPY . /opt/repo

RUN pip install --no-cache /opt/repo