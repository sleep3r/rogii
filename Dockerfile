FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh -o install-uv.sh \
    && sh install-uv.sh \
    && rm install-uv.sh

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-cache --no-dev

FROM nvidia/cuda:12.6.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libgomp1 \
    ocl-icd-libopencl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /root/.local /root/.local
COPY --from=build /app/.venv /app/.venv

RUN --mount=type=secret,id=CLEAR_ML_CONF_PATH,dst=/tmp/clearml.conf,required=false \
    if [ -f /tmp/clearml.conf ]; then cp /tmp/clearml.conf /root/clearml.conf; fi

COPY . .

ARG CMD_ARGS=""
ENV CMD_ARGS=$CMD_ARGS

CMD ["sh", "-c", "uv run python -m rogii $CMD_ARGS"]
