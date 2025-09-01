# ~/pred-svc/Dockerfile
# 基础镜像建议继续用 gcr 缓存，避免直连 Docker Hub
FROM mirror.gcr.io/library/python:3.11-slim

# 安装 ffprobe/ffmpeg（用于媒体特征抽取）与证书
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://|https://|g' /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg ca-certificates; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY listen.py /app/listen.py

# 依赖：FastAPI + Uvicorn + pydantic + 可选 ffmpeg-python
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic ffmpeg-python

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
# 0.0.0.0 监听容器内所有网卡，便于通过 Service / port-forward 访问
CMD ["uvicorn", "listen:app", "--host", "0.0.0.0", "--port", "8000"]
