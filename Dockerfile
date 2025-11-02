# Lightweight dev/runtime image for QuantFlow
FROM mambaorg/micromamba:1.5.10

# Create env via environment.yml
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n quantflow -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Setup workdir
WORKDIR /app
COPY . /app

# Use the env by default
SHELL ["/bin/bash", "-lc"]
ENV MAMBA_DOCKERFILE_ACTIVATE=1
RUN echo "conda activate quantflow" >> ~/.bashrc

# Expose Jupyter port
EXPOSE 8888

# Default command: start JupyterLab
CMD ["bash", "-lc", "conda activate quantflow && jupyter lab --NotebookApp.token='' --NotebookApp.password='' --ip=0.0.0.0 --port=8888 --no-browser"]
