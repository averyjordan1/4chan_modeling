# Use an official Python runtime as a parent image
FROM ubuntu:16.04

# Optional args
ARG python_version=3.6
ARG uid=1000
ARG gid=1000
ARG ostype=Linux

# Configurable settings
ENV USER_NAME pyromancer
ENV CONDA_DIR /opt/conda
ENV WORK_DIR /app
ENV PATH ${CONDA_DIR}/bin:${PATH}

# Install linux utils
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Change to default user
RUN bash -c 'if [ ${ostype} == Linux ]; then groupadd -r --gid ${gid} ${USER_NAME}; fi && \
    useradd -r --create-home --shell /bin/bash --uid ${uid} --gid ${gid} ${USER_NAME}' && \
    mkdir -p ${CONDA_DIR} ${WORK_DIR} && chown ${USER_NAME} ${CONDA_DIR} ${WORK_DIR}
USER ${USER_NAME}

# Install conda
RUN curl -o ~/miniconda.sh -O \
    https://repo.continuum.io/miniconda/Miniconda${python_version%%.*}-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -f -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r app/requirements.txt
RUN pip install jupyter -U && pip install jupyterlab
RUN conda install gensim
# Make port 80 available to the world outside this container
EXPOSE 8888

# Define environment variable
ENV NAME World

# Run app.py when the container launches
ENTRYPOINT ["jupyter", "lab","--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
