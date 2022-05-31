FROM python:3.9

COPY docker/requirements.txt requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential\
         libsm6 \
         libxext6 \
         libxrender-dev \
         ffmpeg  \
    python3-pip &&\
    pip install --upgrade pip  

RUN pip install -r requirements.txt

WORKDIR /app/work/
COPY src src
RUN mkdir /Volumes
RUN mkdir /Volumes/RAMDisk
RUN mount -t tmpfs -o rw,size=2G tmpfs /Volumes/RAMDisk

RUN jupyter nbconvert --to python src/frontend.ipynb

CMD ["bash"]