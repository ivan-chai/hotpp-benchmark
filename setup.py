import setuptools


setuptools.setup(
    name="esp-horizon",
    version="0.0.1",
    author="Ivan Karpukhin",
    author_email="karpuhini@yandex.ru",
    description="Evaluate generative event sequence models on the horizon prediction task.",
    packages=setuptools.find_packages(include=["esp_horizon", "esp_horizon.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "datasets",
        "easy-tpp",
        "hydra-core>=1.1.2",
        "lightgbm",
        "numpy>=1.23",
        "ptls-validation @ git+https://git@github.com/dllllb/ptls-validation.git#egg=ptls-validation",
        "pyarrow>=14.0.0",
        "pyspark>=3",
        "pytorch-lifestream>=0.6.0",
        "pytorch-lightning",
        "scikit-learn>=1.3.2",
        "scipy>=1.11",
        "torch-linear-assignment @ git+https://github.com/ivan-chai/torch-linear-assignment.git#egg=torch-linear-assignment",
        "tqdm",
    ],
)
