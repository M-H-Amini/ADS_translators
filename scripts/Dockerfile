FROM tensorflow/tensorflow:latest

COPY requirements.txt /usr/src/app

WORKDIR /usr/src/app

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx