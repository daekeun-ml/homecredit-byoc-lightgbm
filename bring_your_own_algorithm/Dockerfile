FROM ubuntu:18.04
 
RUN apt -y update && apt install -y --no-install-recommends \
    wget \
    python3-distutils \
    nginx \
    ca-certificates \
    libgomp1 \
    && apt clean
 

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
    pip install wheel numpy scipy scikit-learn pandas lightgbm flask gevent gunicorn && \
    rm -rf /root/.cache

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
 
COPY lightgbm /opt/program
 
WORKDIR /opt/program

# Entrypoint script comes from sagemaker_container_support
ENTRYPOINT ["/usr/bin/python"]