FROM ghcr.io/coreweave/nccl-tests:12.3.1-devel-ubuntu22.04-nccl2.19.3-1-a72ab6c
LABEL authors="Alex Kharlamov"

RUN apt update && apt install -y libibverbs-dev \
    rdma-core \
    htop \
    nload \
    ncdu \
    ffmpeg \
    libsm6  \
    libxext6 \
    apt-transport-https \
    ca-certificates \
    curl \
    unzip \
    python3-pip \
    git

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch packaging

RUN pip install ninja -U && pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

RUN export MAX_JOBS=3 && git clone https://github.com/Dao-AILab/flash-attention && \
    cd flash-attention && \
    python setup.py install && \
    cd csrc/rotary && pip install . && \
    cd ../layer_norm && pip install . && \
    cd ../xentropy && pip install . && \
    cd ../.. && rm -rf flash-attention

COPY requirements.txt /container/requirements.txt
RUN pip install -r /container/requirements.txt

COPY . /genf/