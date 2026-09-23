# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

FROM node:22-bookworm-slim AS ui-build

WORKDIR /ui
COPY package.json package-lock.json ./
RUN npm ci
COPY frontend ./frontend
RUN npm run build

FROM ${CUDA_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PYOPENGL_PLATFORM=egl \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git git-lfs \
      libegl1 libegl1-mesa-dev libeigen3-dev libgl1 libgles2 libglvnd-dev \
      libglib2.0-0 libgomp1 libsm6 libxext6 libxi6 libxrender1 ninja-build \
      pkg-config python3.10 python3-dev python3-pip python3-venv wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
      torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124

FROM base AS vendor

ARG HUNYUAN_CODE_REV=82920d643c0dc2f7bfd7255f45f62d386edfe60c
ARG SAM2_CODE_REV=2b90b9f5ceec907a1c18123530e92e794ad901a4
ARG MESHOPT_CODE_REV=9d9890c73011d75920af614485296d1e03e95448

RUN git init /opt/vendor/Hunyuan3D-2.1 \
    && git -C /opt/vendor/Hunyuan3D-2.1 remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git \
    && git -C /opt/vendor/Hunyuan3D-2.1 fetch --depth 1 origin ${HUNYUAN_CODE_REV} \
    && git -C /opt/vendor/Hunyuan3D-2.1 checkout --detach FETCH_HEAD \
    && git init /opt/vendor/sam2 \
    && git -C /opt/vendor/sam2 remote add origin https://github.com/facebookresearch/sam2.git \
    && git -C /opt/vendor/sam2 fetch --depth 1 origin ${SAM2_CODE_REV} \
    && git -C /opt/vendor/sam2 checkout --detach FETCH_HEAD \
    && git init /tmp/meshoptimizer \
    && git -C /tmp/meshoptimizer remote add origin https://github.com/zeux/meshoptimizer.git \
    && git -C /tmp/meshoptimizer fetch --depth 1 origin ${MESHOPT_CODE_REV} \
    && git -C /tmp/meshoptimizer checkout --detach FETCH_HEAD \
    && cmake -S /tmp/meshoptimizer -B /tmp/meshoptimizer/build \
         -DMESHOPT_BUILD_GLTFPACK=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/meshoptimizer/build --target gltfpack --parallel \
    && install -m 0755 /tmp/meshoptimizer/build/gltfpack /usr/local/bin/gltfpack \
    && rm -rf /tmp/meshoptimizer

RUN python3 -m pip install -r /opt/vendor/Hunyuan3D-2.1/requirements.txt \
    && python3 -m pip install -e /opt/vendor/sam2 \
    && python3 -m pip install ultralytics

RUN python3 -m pip install -e /opt/vendor/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer \
    && cd /opt/vendor/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer \
    && bash compile_mesh_painter.sh

FROM vendor AS app-test

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY frontend ./frontend
COPY --from=ui-build /ui/frontend/dist ./frontend/dist
RUN python3 -m pip install -e '.[dev]' \
    && ruff check . \
    && mypy src \
    && pytest

FROM vendor AS model-bake

ARG HUNYUAN_MODEL_REV=0b94677654c57bb9a6b6845cd7b704ccf551d327
ARG DINOV2_MODEL_REV=611a9d42f2335e0f921f1e313ad3c1b7178d206d
ARG YOLO_MODEL_REPO
ARG YOLO_MODEL_REV

ENV HF_HOME=/opt/models/huggingface \
    HF_HUB_CACHE=/opt/models/huggingface/hub

COPY scripts/download_models.py /usr/local/bin/download_models.py
COPY scripts/validate_templates.py /usr/local/bin/validate_templates.py
COPY templates /opt/templates
RUN --mount=type=secret,id=hf_token,required=false \
    test -n "${YOLO_MODEL_REPO}" \
    && test -n "${YOLO_MODEL_REV}" \
    && HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
       python3 /usr/local/bin/download_models.py \
         --hunyuan-revision "${HUNYUAN_MODEL_REV}" \
         --dinov2-revision "${DINOV2_MODEL_REV}" \
         --yolo-repo "${YOLO_MODEL_REPO}" \
         --yolo-revision "${YOLO_MODEL_REV}" \
         --sam-config /opt/vendor/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
    && python3 /usr/local/bin/validate_templates.py /opt/templates

FROM vendor AS final

ENV APP_ENV=production \
    PIPELINE_BACKEND=gpu \
    STORAGE_BACKEND=r2 \
    MODEL_ROOT=/opt/models \
    MODEL_MANIFEST_PATH=/opt/models/model-manifest.json \
    HF_HOME=/opt/models/huggingface \
    HF_HUB_CACHE=/opt/models/huggingface/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DIFFUSERS_OFFLINE=1 \
    PYTHONPATH=/opt/vendor/Hunyuan3D-2.1/hy3dpaint

WORKDIR /app
# Keep large model families in separate OCI layers. This improves registry caching and
# avoids producing one oversized layer for the complete baked model set.
COPY --from=model-bake /opt/models/huggingface/hub/models--tencent--Hunyuan3D-2.1 /opt/models/huggingface/hub/models--tencent--Hunyuan3D-2.1
COPY --from=model-bake /opt/models/huggingface/hub/models--facebook--dinov2-giant /opt/models/huggingface/hub/models--facebook--dinov2-giant
COPY --from=model-bake /opt/models/yolo /opt/models/yolo
COPY --from=model-bake /opt/models/sam2 /opt/models/sam2
COPY --from=model-bake /opt/models/realesrgan /opt/models/realesrgan
COPY --from=model-bake /opt/models/model-manifest.json /opt/models/model-manifest.json
COPY --from=model-bake /opt/templates /opt/templates
COPY --from=ui-build /ui/frontend/dist /app/ui
COPY pyproject.toml README.md ./
COPY src ./src
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN python3 -m pip install . \
    && chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /tmp/eyewear-vto/work

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "eyewear_vto.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
