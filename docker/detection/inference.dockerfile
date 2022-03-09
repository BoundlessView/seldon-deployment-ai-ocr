# syntax=docker/dockerfile:1.2
FROM seldonio/seldon-core-s2i-python36:1.1.0
ARG EXPERIMENT_NAME
WORKDIR /app

RUN python3.6 -m pip install pip==19
RUN pip install opencv-python-headless==4.2.0.34 \
                intel_tensorflow==1.14.0 \
                gast==0.2.2 \
                pyyaml \
                git+https://github.com/tensorpack/tensorpack.git@v0.9.8

COPY models/$EXPERIMENT_NAME/compact.pb /app/compact.pb
COPY models/$EXPERIMENT_NAME/config.yaml /app/config.yaml
COPY src/inference/$EXPERIMENT_NAME/* /app/

CMD seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE

