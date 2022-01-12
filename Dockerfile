#specify base image
FROM python:3.8-slim-buster

#add metadata
LABEL maintainer = "Sam Mfalila <sam.mfalila@gmail.com>"

#create work directory
RUN mkdir /app

#navigate to work dir
WORKDIR /app

#create work dir
RUN mkdir /app/data

#navigate to work dir
WORKDIR /app/data

#create work dir
RUN mkdir /app/data/processed

#navigate to work dir
WORKDIR /app/data/processed

#copy files
copy ["/data/processed/preprocessing_pipeline.joblib", "./"]

#exit dir
WORKDIR ../../

#create work dir
RUN mkdir /app/build_library

#navigate to work dir
WORKDIR /app/build_library

#copy files
copy ["/build_library", "./"]

#exit dir
WORKDIR ../

#create work directory
RUN mkdir /app/models

#navigate to work dir
WORKDIR /app/models

#copy files
copy ["models/clf_model_weights.h5", "./"]

#exit dir
WORKDIR ../

#create work directory
RUN mkdir /app/webapp

#navigate to work dir
WORKDIR /app/webapp

#copy files
copy ["/webapp", "./"]

#exit dir
WORKDIR ../

#create work directory
RUN mkdir /app/src

#navigate to work dir
WORKDIR /app/src

#create work dir
RUN mkdir /app/src/data

#navigate to work dir
WORKDIR /app/src/data

#copy files
copy ["/src/data/load_data.py", "./"]

#exit dir
WORKDIR ../../

#create work directory
RUN mkdir /app/prediction_service

#navigate to work dir
WORKDIR /app/prediction_service

#create work directory
RUN mkdir /app/prediction_service/model

#navigate to work dir
WORKDIR /app/prediction_service/model

#create work directory
RUN mkdir /app/prediction_service/model/tfmodel

#navigate to work dir
WORKDIR /app/prediction_service/model/tfmodel

#copy files
copy ["/prediction_service/model/tfmodel/clf_checkpoint.joblib",\
"/prediction_service/model/tfmodel/clf_model.json", \
"/prediction_service/model/tfmodel/clf_model_weights.h5", "./"]

#exit dir
WORKDIR ../../

#copy files
copy ["/prediction_service/__init__.py","/prediction_service/prediction.py","/prediction_service/schema_in.json", "./"]

#exit dir
WORKDIR ../

#navigate to work dir
WORKDIR /app

#copy all files
COPY ["setup.py", "params.yaml", "app.py", "Procfile", "requirements.txt", "./"]


#install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

#run
CMD ["python3","./app.py"]

