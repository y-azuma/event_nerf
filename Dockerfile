ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY event_nerf /workspace/event_nerf
RUN python -m pip install --no-cache-dir -e ".[eval,dev,wandb]"

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg
ENTRYPOINT ["python", "-m"]
