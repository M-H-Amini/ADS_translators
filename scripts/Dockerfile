FROM tensorflow/tensorflow:latest

WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx