#specify base image
# FROM ubuntu:20.04
FROM python:3.8-slim-buster

#ENV VIRTUAL_ENV=/opt/bmve/backorder_analytics_mlops
#RUN python3 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"

#install python
# RUN apt-get -y update && apt-get install python3 -y && \
#    apt-get --no-install-recommends -y install python3-pip -y

#add metadata
LABEL maintainer = "Sam Mfalila <sam.mfalila@gmail.com>"

#create work directory
RUN mkdir /app

#navigate to work directory
WORKDIR /app

# ENV PYTHONPATH="${PYTHONPATH}:/app"
# RUN python3 -m venv venv
#copy all files
COPY ["setup.py", "params.yaml", "webapp/static/css/main.css","webapp/templates/404.html", "webapp/templates/base.html", "webapp/templates/index.html", "app.py", "prediction_service/model/tfmodel/clf_checkpoint.joblib", \
  "prediction_service/model/tfmodel/clf_model_weights.h5","prediction_service/model/tfmodel/clf_model.json",\
  "Procfile", "requirements.txt", "./"]

# ENV PYTHONPATH="${PYTHONPATH}:/app"
# ENV PYTHONPATH "${PYTHONPATH}:/mnt/e/venvs/backorder_analytics_mlops/bmve/backorder_analytics_mlops"

#install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# add PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

#run
CMD python3 app.py
