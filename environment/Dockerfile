# hash:sha256:c2b9490db7125055e66d874328b4ad37296ca7ccd1dbda75eeeb503f1a8ac52b
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-log-utils==0.2.3 \
    imbalanced-learn==0.11.0 \
    lightgbm==4.1.0 \
    mlxtend==0.23.0 \
    pyvips==2.2.1 \
    scikit-optimize==0.9.0 \
    seaborn==0.13.0 \
    spikeinterface[full]==0.101.2 \
    umap-learn==0.5.4 \
    wavpack-numcodecs==0.2.1 \
    xgboost==2.0.2

COPY postInstall /
RUN /postInstall
