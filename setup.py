import os
import setuptools

with open("README.md") as fp:
    long_description = fp.read()

extras_kwargs = {}

if not os.environ.get("HOTPP_PUBLISH", False):
    # PyPI doesn't support direct links.
    extras_kwargs["extras_require"] = {
        "downstream":  ["ptls-validation @ git+https://git@github.com/dllllb/ptls-validation.git#egg=ptls-validation"]
    }


setuptools.setup(
    name="hotpp-benchmark",
    version="0.6.4",
    author="Ivan Karpukhin",
    author_email="karpuhini@yandex.ru",
    description="Evaluate generative event sequence models on the long horizon prediction task.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=["hotpp", "hotpp.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "datasets",
        "hydra-core>=1.1.2",
        "lightgbm",
        "numpy>=1.23",
        "pyarrow>=14.0.0",
        "pymonad",
        "pyspark>=3",
        "pytorch-lifestream>=0.6.0",
        "pytorch-lightning",
        "scikit-learn>=1.3.2",
        "scipy>=1.11",
        "torch-linear-assignment",
        "tqdm",
    ],
    **extras_kwargs
)
