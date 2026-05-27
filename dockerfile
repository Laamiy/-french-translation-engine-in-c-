# ==========================================
# Builder Stage
# ==========================================
FROM ubuntu:24.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y curl && curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz -o onnxruntime.tgz \
    && tar -xzf onnxruntime.tgz --strip-components=1 -C /usr/local \
    && rm onnxruntime.tgz

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libdrogon-dev \
    libjsoncpp-dev \
    libsentencepiece-dev \
    libsqlite3-dev \
    libpq-dev \
    libmysqlclient-dev \
    libbrotli-dev \
    libhiredis-dev \
    libyaml-cpp-dev \
    libprotobuf-dev \
    protobuf-compiler \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*


COPY ./src /app/src
COPY ./CMakeLists.txt /app/CMakeLists.txt
COPY ./Thirdparty /app/Thirdparty

RUN cmake -S /app -B /app/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /app/build --config Release

# ==========================================
# Runtime Stage
# ==========================================
FROM ubuntu:24.04 as runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libjsoncpp25 \
    libsqlite3-0 \
    libpq5 \
    libmysqlclient21 \
    libbrotli1 \
    libhiredis1.1.0 \
    libyaml-cpp0.8 \
    libprotobuf32t64 \
    libc-ares2 \
    libssl3 \
    zlib1g \
    libuuid1 \
    && rm -rf /var/lib/apt/lists/*


# ONNX Runtime shared library 
COPY --from=builder /usr/local/lib/libonnxruntime.so* /usr/local/lib/

# Drogon shared library 
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libdrogon.so* /usr/local/lib/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libtrantor.so* /usr/local/lib/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libsentencepiece.so* /usr/local/lib/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libmariadb.so* /usr/local/lib/
# SentencePiece shared library 
COPY --from=builder /usr/lib/x86_64-linux-gnu/libsentencepiece.so* /usr/local/lib/

# Refresh linker 
RUN ldconfig

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# configuration and models
COPY --chown=appuser:appuser ./config.xml /app/config.xml
COPY --chown=appuser:appuser ./models/onnx-en-fr-q /app/models/onnx-en-fr-q

# application executable
COPY --from=builder --chown=appuser:appuser /app/build/translation_server /app/translation_server

EXPOSE 8000

CMD ["/app/translation_server", "/app/config.xml"]