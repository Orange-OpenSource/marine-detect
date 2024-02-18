# Use a base image with Python and Pip installed
# https://hub.docker.com/_/python
FROM python:3.10-slim-bullseye

# Base image environment variable(s)
# Allow statements and log messages to be sent straight to the terminal without being buffered
ENV PYTHONUNBUFFERED True

# Use the root user
USER root

# Update pip
RUN pip install --upgrade pip

# Set the working directory
WORKDIR /tmp

# Install the required development dependencies
RUN apt-get update &&\
    apt-get install -y \
    git \
    ffmpeg \ 
    libsm6 \ 
    libxext6 &&\
    # Cleaning to keep the layer size minimum
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*

# Install the Python development environment
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
