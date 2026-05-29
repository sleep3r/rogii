# RAC-Former server overlay image.
#
# `spacebridge train` always runs `docker build` before every remote launch.
# During normal code iteration we do not want that build to reinstall apt,
# Python, Torch, and CUDA wheels.  This Dockerfile therefore starts from the
# last full RAC-Former image and only overlays the current source tree.
#
# If dependencies changed, rebuild the base with:
#     make docker-build-full

ARG RACFORMER_BASE_IMAGE=harbor.wildberries.ru/tns-backoffice-security/spacebrigde:racformer-base-aleksandr-kalashnikov
FROM ${RACFORMER_BASE_IMAGE}

WORKDIR /app

# Optional ClearML config (mounted as build secret by spacebridge).
RUN --mount=type=secret,id=CLEAR_ML_CONF_PATH,dst=/tmp/clearml.conf,required=false \
    if [ -f /tmp/clearml.conf ]; then cp /tmp/clearml.conf /root/clearml.conf; fi

# Replace source atomically so removed files from the previous image do not linger.
RUN rm -rf /app/racformer
COPY . ./racformer/

ARG CMD_ARGS="racformer/configs/racformer_sanity.yml"
ENV CMD_ARGS=$CMD_ARGS

CMD ["sh", "-c", "python -m racformer.train ${CMD_ARGS#--config=}"]
