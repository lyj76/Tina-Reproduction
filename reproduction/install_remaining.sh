#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Tina
source reproduction/env.sh
env -u http_proxy -u https_proxy -u ALL_PROXY uv pip install --python "$(command -v python)" --index-url https://mirrors.aliyun.com/pypi/simple 'setuptools<81'
env -u http_proxy -u https_proxy -u ALL_PROXY uv pip install --python "$(command -v python)" --index-url https://download.pytorch.org/whl/cu118 \
  --no-deps xformers==0.0.28.post3
env -u http_proxy -u https_proxy -u ALL_PROXY uv pip install --python "$(command -v python)" --no-build-isolation \
  --index-url https://mirrors.aliyun.com/pypi/simple -c reproduction/constraints.txt \
  -r reproduction/requirements.txt \
  '/root/autodl-tmp/vllm-0.7.2+cu118-cp38-abi3-manylinux1_x86_64.whl' \
  '/root/autodl-tmp/flash_attn-2.7.3+cu11torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
