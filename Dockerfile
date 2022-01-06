#specify base image
FROM ubuntu:20.04

#install python
RUN apt-get -y update && apt-get install python3 -y && \
    apt-get --no-install-recommends -y install python3-pip -y

#add metadata
LABEL maintainer = "Sam Mfalila <sam.mfalila@gmail.com>"

#create work directory
RUN mkdir /app

#navigate to work directory
WORKDIR /app

#copy all files
COPY ["setup.py", "params.yaml", "webapp", "build_library", "src/data/load_data.py", "app.py", "prediction_service/prediction.py", \
      "prediction_service/model/tfmodel/clf_checkpoint.joblib", "prediction_service/model/tfmodel/clf_model_weights.h5", \
       "prediction_service/model/tfmodel/clf_model.json", "Procfile", "requirements.txt", "./"]

#install dependencies
RUN python3 -m  pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt --no-cache-dir

#run
CMD python3 app.py
