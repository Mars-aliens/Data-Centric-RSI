ARG PYTORCH_BASE_IMAGE
FROM ${PYTORCH_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/candidate \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /candidate
COPY candidate-requirements.txt /tmp/candidate-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/candidate-requirements.txt \
    && python -m pip check \
    && python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"

USER 65532:65532
