uv venv --python=python3.11
source .venv/bin/activate

uv pip install -e models/qwen2-5-vl
uv pip install -e models/qwen-vl-utils

uv pip install ffmpeg-python==0.2.0 moviepy==1.0.3   # for StreamingBench / OVO-Bench

uv pip install torch==2.8.0 torchvision==0.23.0
uv pip install transformers==4.49
uv pip install flash_attn --no-build-isolation

uv pip install decord==0.6.0