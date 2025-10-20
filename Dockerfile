# ==================================================================================================
# АРГУМЕНТЫ СБОРКИ
# ==================================================================================================
ARG WHISPER_MODEL="large-v3"
ARG ALIGN_LANG="ru"
ARG HF_TOKEN=""
ARG UID=1001

# ==================================================================================================
# БАЗОВЫЙ ОБРАЗ
# ==================================================================================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/torch/lib/../../nvidia/cudnn/lib/:/opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        git \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

# ==================================================================================================
# BUILDER - УСТАНОВКА ЗАВИСИМОСТЕЙ
# ==================================================================================================
FROM base AS builder

WORKDIR /build

# Установка uv
ADD https://astral.sh/uv/install.sh /tmp/uv-install.sh
RUN sh /tmp/uv-install.sh && rm /tmp/uv-install.sh

ENV PATH="/root/.local/bin:$PATH"

# Копируем файлы зависимостей
COPY pyproject.toml uv.lock* ./

# Создаем venv и устанавливаем зависимости
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache -r pyproject.toml

# Копируем и устанавливаем сам проект
COPY . .
RUN . /opt/venv/bin/activate && \
    uv pip install --no-cache .

# ==================================================================================================
# RUNTIME БЕЗ МОДЕЛЕЙ
# ==================================================================================================
FROM base AS runtime-base

ARG UID

# Создаем пользователя и директории
RUN groupadd -g ${UID} appuser && \
    useradd -l -u ${UID} -g appuser -m -s /bin/bash appuser && \
    mkdir -p /.cache /app && \
    chown -R appuser:appuser /.cache /app

# Копируем ffmpeg и dumb-init
COPY --from=ghcr.io/jim60105/static-ffmpeg-upx:8.0 /ffmpeg /usr/local/bin/
COPY --from=ghcr.io/jim60105/static-ffmpeg-upx:8.0 /dumb-init /usr/local/bin/

# Копируем виртуальное окружение
COPY --from=builder /opt/venv /opt/venv

# Настройка окружения
ENV PATH="/opt/venv/bin:$PATH" \
    XDG_CACHE_HOME="/.cache" \
    TORCH_HOME="/.cache/torch" \
    HF_HOME="/.cache/huggingface" \
    NUMBA_CACHE_DIR="/.cache/numba"

ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/torch/lib/../../nvidia/cudnn/lib/:/opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH
USER root
RUN set -eux; \
    venv_cudnn="/opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib"; \
    if [ -d "$venv_cudnn" ]; then \
        cd "$venv_cudnn"; \
        ln -sf libcudnn_cnn.so.9 libcudnn_cnn.so.9.1.0 || true; \
        ln -sf libcudnn_cnn.so.9 libcudnn_cnn.so.9.1 || true; \
        ln -sf libcudnn_cnn.so.9 libcudnn_cnn.so || true; \
    fi; \
    ldconfig || true

WORKDIR /app
USER appuser

ENTRYPOINT ["dumb-init", "--", "whisperx"]
CMD ["--help"]

# ==================================================================================================
# FINAL - С ПРЕДЗАГРУЖЕННЫМИ МОДЕЛЯМИ
# ==================================================================================================
FROM runtime-base AS final

USER appuser

ARG WHISPER_MODEL
ARG ALIGN_LANG
ARG HF_TOKEN

# Предзагрузка Silero VAD
RUN python -c "\
import torch; \
torch.hub.load(\
    repo_or_dir='snakers4/silero-vad', \
    model='silero_vad', \
    force_reload=False, \
    onnx=False, \
    trust_repo=True\
)" && echo "✓ Silero VAD loaded"

# Предзагрузка Whisper
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel(\
    '${WHISPER_MODEL}', \
    device='cpu', \
    compute_type='int8'\
)" && echo "✓ Whisper ${WHISPER_MODEL} loaded"

# Предзагрузка модели выравнивания
RUN python -c "\
from whisperx.alignment import load_align_model; \
load_align_model('${ALIGN_LANG}', 'cpu')\
" && echo "✓ Alignment model (${ALIGN_LANG}) loaded"

# Предзагрузка диаризации (если токен предоставлен)
RUN if [ -n "${HF_TOKEN}" ]; then \
        python -c "\
from pyannote.audio import Pipeline; \
Pipeline.from_pretrained(\
    'pyannote/speaker-diarization-3.1', \
    use_auth_token='${HF_TOKEN}'\
)"; \
        echo "✓ Diarization model loaded"; \
    else \
        echo "⚠ HF_TOKEN not set, skipping diarization"; \
    fi

# Проверка установки
RUN whisperx --help > /dev/null && echo "✓ WhisperX ready"
