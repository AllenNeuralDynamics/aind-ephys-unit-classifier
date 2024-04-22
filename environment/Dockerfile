# hash:sha256:07e562da92ad598381fc7ee0ed7c3b5694f9a69706622b522f67d35533dff9e7
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUSTOM_KEY
ARG CUSTOM_KEY_2
ARG CUSTOM_KEY_3

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.19.1 \
    imbalanced-learn==0.11.0 \
    lightgbm==4.1.0 \
    mlxtend==0.23.0 \
    pyvips==2.2.1 \
    scikit-optimize==0.9.0 \
    seaborn==0.13.0 \
    umap-learn==0.5.4 \
    wavpack-numcodecs==0.1.5 \
    xgboost==2.0.2

COPY postInstall /
RUN /postInstall
