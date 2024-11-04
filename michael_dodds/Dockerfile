FROM python:3.10

WORKDIR /app

COPY requirements.txt  /app/requirements.txt

VOLUME /app/working_files

EXPOSE 1222

USER root

RUN python3 -m pip install -r requirements.txt

CMD ["/bin/bash"]
