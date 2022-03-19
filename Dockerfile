# =================== Builder stage ======================
# Set up a pipenv builder first, then copy its environment
FROM python:3.8.12-slim as builder

ENV PATH="/root/.local/bin:/usr/src/.venv/bin:${PATH}"

RUN pip install --user pipenv
# Tell pipenv to create venv in the current directory
ENV PIPENV_VENV_IN_PROJECT=1
# Pipefile contains requests
ADD Pipfile.lock Pipfile /usr/src/

WORKDIR /usr/src

RUN pipenv sync

# Testing that python works
RUN python -c "import requests; print(requests.__version__)"


# =================== Runtime stage ======================
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04 as runtime

# Bring pipenv environment to the cuda container
RUN mkdir -v /usr/src/venv
COPY --from=builder /usr/src/.venv/ /usr/src/venv/
ENV PATH="/usr/src/venv/bin:${PATH}"

# We need to install python anyways to be able to use it
RUN apt-get update && apt-get install -y --no-install-recommends \
    # These are for building python
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    wget \
    # and these are for project support
    ffmpeg \
    libsm6 \
    libxext6 \
    libbz2-dev \
    liblzma-dev

RUN mkdir -p tmp/
WORKDIR tmp/
RUN wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz
RUN tar -xvzf Python-3.8.12.tgz
WORKDIR Python-3.8.12
RUN ./configure
RUN make install

# Testing that python works
RUN python -c "import requests; print(requests.__version__)"
# Testing that installed thirdparty library works too
RUN python -c "import cv2; print(cv2.__version__)"

# these two lines are mandatory for cloud training with Grid AI
WORKDIR /gridai/project
COPY . .
