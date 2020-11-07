FROM ros:melodic-perception

WORKDIR /app

# Conda
RUN apt-get update
RUN apt-get install curl libcap-dev

ENV PATH /opt/conda/bin:$PATH

# From https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

COPY conda-env.yml .
COPY requirements.txt .
RUN conda env create -f conda-env.yml

RUN echo 'conda activate vilbert-mt' >> ~/.bashrc

# OpenGL + GPU
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN apt-get install -y terminator dbus ros-melodic-rviz ros-melodic-vision-opencv

# TODO remove
RUN conda run -n vilbert-mt pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.3.1/en_core_web_md-2.3.1.tar.gz