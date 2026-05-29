ARG BPHWT_BASE_IMAGE=harbor.wildberries.ru/tns-backoffice-security/spacebrigde:bphwt-base-aleksandr-kalashnikov
FROM ${BPHWT_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/.venv/bin:/root/.local/bin:$PATH"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN --mount=type=secret,id=CLEAR_ML_CONF_PATH,dst=/tmp/clearml.conf,required=false \
    if [ -f /tmp/clearml.conf ]; then cp /tmp/clearml.conf /root/clearml.conf; fi

COPY . .

ARG CMD_ARGS=""
ENV CMD_ARGS=$CMD_ARGS
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "python -m bphwt.docker_entrypoint $CMD_ARGS"]
