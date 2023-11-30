FROM ubuntu:22.04

# Install miniconda
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh

# Install required conda packages
RUN conda init
RUN conda create --name langchain-gpu python=3.11
RUN conda activate langchain-gpu
RUN conda install langchain -c conda-forge
RUN conda install -c huggingface transformers
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install ipykernel
RUN conda install -n lc-update ipykernel --update-deps --force-reinstall

# Install required package extensions using pip
RUN pip install sentence_transformers
RUN pip install accelerate
RUN pip install optimum
RUN pip install auto-gptq
RUN pip install chromadb

# Clone from git directory
# RUN git clone <url>