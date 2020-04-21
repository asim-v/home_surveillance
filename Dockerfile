FROM mdegans/tegra-opencv:latest
MAINTAINER domcross

### environment variables ###
# required for apt-get -y to work properly:
ENV DEBIAN_FRONTEND=noninteractive

ARG USER_HOMEDIR="/var/home_surveillance"

# install python build deps, install python packages, purge build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        cuda-compiler-10-0 \
        cuda-libraries-dev-10-0 \
        cuda-minimal-build-10-0 \
        git \
        libcudnn7-dev \
        libopenmpi-dev \
        libopenmpi2 \
        python3 \
        python3-dev \
        python3-pip \
        wget \
        gfortran \
        libfreetype6-dev

# pytorch 1.4 and more python packages
RUN wget -nv --show-progress --https-only --progress=bar:force:noscroll https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
RUN python3 -m pip install -U pip && python3 -m pip install Cython setuptools && python3 -m pip install -U numpy scipy && \
    python3 -m pip install torch-1.4.0-cp36-cp36m-linux_aarch64.whl
RUN python3 -m pip install matplotlib pandas>=1.0.2 requests>=2.23.0 psutil>=5.7.0 scikit-learn scipy Werkzeug==0.16.1 websocket-client apprise Flask==0.11.1 Flask-Uploads==0.2.1 Flask-SocketIO==2.5 websocket-client apprise Flask-Uploads==0.2.1 attrs==19.1.0

RUN adduser --system --group --home ${USER_HOMEDIR} home_surveillance \
    && git clone https://github.com/davisking/dlib.git \
    && cd dlib \
    && python3 setup.py install --set DLIB_USE_CUDA=1 \
    && cd ${USER_HOMEDIR} \
    && rm -rf dlib \
    && git clone https://github.com/cmusatyalab/openface \
    && cd openface \
    && python3 setup.py install \
    && cd ${USER_HOMEDIR} \
    && rm -rf openface \
    && chmod -R 755 ${USER_HOMEDIR} \
    && sudo apt-get purge -y --autoremove \
        build-essential \
        cmake \
        cuda-compiler-10-0 \
        cuda-libraries-dev-10-0 \
        cuda-minimal-build-10-0 \
        git \
        libcudnn7-dev \
        libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# drop to user and cd ~
USER home_surveillance:home_surveillance
WORKDIR ${USER_HOMEDIR}

# copy last, for easy changes during development
COPY --chown=home_surveillance:home_surveillance . ${USER_HOMEDIR}
RUN mkdir aligned-images \
    && mkdir training-images

EXPOSE 5000
# ENTRYPOINT ["python3", "system/WebApp.py"]
