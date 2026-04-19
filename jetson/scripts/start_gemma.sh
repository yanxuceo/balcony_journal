#!/bin/bash
sudo docker run -it --rm --pull missing --runtime=nvidia --network host \
  -v /home/xudong/balcony_journal/models/hf_cache:/data/models/huggingface \
  ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin \
  llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S
