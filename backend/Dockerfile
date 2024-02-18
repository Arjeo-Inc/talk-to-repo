FROM python:3.12
ENV VIRTUAL_ENV=/usr/local
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

# Install some basic utilities.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    libssl-dev \
 && rm -rf /var/lib/apt/lists/*

 # Create a working directory.
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN /root/.cargo/bin/uv pip install --no-cache -r requirements.txt
COPY . /app/

EXPOSE 8000

CMD ["gunicorn", "main:app", "--bind", "[::]:8000", "-k", "uvicorn.workers.UvicornWorker"]
