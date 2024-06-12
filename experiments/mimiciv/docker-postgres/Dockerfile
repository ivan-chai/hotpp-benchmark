FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PG_MAJOR_VER=14

RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC+3

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:apt-fast/stable --yes
RUN apt update && apt-get install -y apt-fast
RUN apt-get update && apt-fast install -y \
    libopenblas-dev \
    libssl-dev \
    bison \
    flex \
    pkg-config \
    libreadline-dev \
    libz-dev \
    curl \
    lsb-release \
    tzdata \
    sudo \
    cmake \
    libpq-dev \
    libclang-dev \
    wget \
    postgresql-plpython3-$PG_MAJOR_VER \
    postgresql-$PG_MAJOR_VER \
    postgresql-server-dev-$PG_MAJOR_VER \
    git

WORKDIR /app

# add postgres user to sudoers
RUN echo 'postgres ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

RUN chmod a+rwx `$(which pg_config) --pkglibdir` \
    `$(which pg_config) --sharedir`/extension \
    /var/run/postgresql/

RUN sudo chown -R postgres:postgres /app

ENV PGDATA /var/lib/postgresql/$PG_MAJOR_VER/main/pgdata

RUN pip install --upgrade pip

USER postgres
WORKDIR /var/lib/postgresql

RUN git clone https://github.com/MIT-LCP/mimic-code.git
RUN git clone https://github.com/mmcdermott/EventStreamGPT.git
RUN git clone https://github.com/mmcdermott/MIMICIV_FMs_public.git

COPY --chown=postgres:postgres mimic.env MIMICIV_FMs_public/.env
COPY --chown=postgres:postgres dataset.yaml MIMICIV_FMs_public/configs/dataset.yaml
COPY --chown=postgres:postgres createdb.sh createdb.sh
RUN chmod +x createdb.sh

ENV PATH="/var/lib/postgresql/.local/bin:${PATH}"

RUN cd EventStreamGPT && pip install -e .

RUN pip install rootutils connectorx>=0.3.1
