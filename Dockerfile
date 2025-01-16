FROM dataloopai/dtlpy-agent:cpu.py3.10.pytorch2

USER 1000
WORKDIR /tmp/app
ENV HOME=/tmp

COPY . /tmp/app
RUN pip install --user \
    dtlpy \
    "pyyaml>=5.3.1" \
    onnxruntime \
    ftfy \
    regex \
    'pillow>=11.0.0' \
    git+https://github.com/openai/CLIP.git
ENV PYTHONPATH=/tmp/.local/bin:/tmp/app/3rd_party:/tmp/app

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/yolo-world-sam-toolbar:0.1.8 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/yolo-world-sam-toolbar:0.1.8
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/yolo-world-sam-toolbar:0.1.6 bash