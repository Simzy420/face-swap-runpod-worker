FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r /requirements.txt

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
