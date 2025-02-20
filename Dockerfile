FROM debian:bullseye

ENV http_proxy=http://proxy.cse.cuhk.edu.hk:8000
ENV https_proxy=http://proxy.cse.cuhk.edu.hk:8000

RUN apt update \
  && apt install build-essential -y \
  && apt install libboost-all-dev -y \
  && apt install libomp-dev -y \
  && apt install libfmt-dev -y \
  && apt install git -y \
  && apt-get clean \
  && cd /home \
  && git clone https://github.com/facebookresearch/faiss.git \
  && git clone https://github.com/nmslib/hnswlib.git