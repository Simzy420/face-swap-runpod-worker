---
title: Face Swap, Extend & Templates
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
---

# Personal creative tools

Templates generation runs on RunPod. Face Swap and Extend are available in
the top navigation and open their existing hosted apps.

Source: https://github.com/Simzy420/face-swap-runpod-worker

Configure `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, `RUNPOD_ENDPOINT_URL`, and
`HF_TOKEN` using Space secrets and variables. No model weights are loaded by
this web container. The default Space hardware is CPU Basic.
