FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

RUN apt-get update -y && \
    apt-get install -y git unzip libblas3 liblapack3 liblapack-dev libblas-dev gfortran libatlas-base-dev \
                       cmake openjdk-8-jdk python3-pip cuda-nvcc-11-7 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "hydra-core==1.3.2" "lightgbm==4.3.0" "numpy==1.26.4" "pyarrow==16.0.0" \
                               "pyspark==3.4.2" "pytest" "scikit-learn==1.4.2" "scipy==1.13.0" "tqdm==4.66.4"
RUN pip install --no-cache-dir "torch==2.0.1+cu117" --index-url https://download.pytorch.org/whl/cu117
RUN pip install --no-cache-dir "pytorch-lightning==2.2.4"
RUN pip install --no-cache-dir "datasets==2.19.0" "transformers==4.40.1" "wandb" "duckdb"

RUN pip install --no-cache-dir "ptls-validation @ git+https://git@github.com/dllllb/ptls-validation.git#egg=ptls-validation"
RUN pip install --no-cache-dir "pytorch-lifestream==0.6.0"

ENV TORCH_CUDA_ARCH_LIST="7.0 8.0 8.6"
RUN pip install --no-build-isolation torch-linear-assignment