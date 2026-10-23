# face-swap-runpod-worker

RunPod serverless worker for Casey's private face swap (`Simzy/face-swap-runpod` Space).
Same pipeline as the ZeroGPU Space `Simzy/bfs-best-face-swap`: Qwen-Image-2.1 + BFS head-swap LoRA
(`Simzy/BFS-Best-Face-Swap`) + optional Viggle 6-step turbo LoRA.

- Image: `ghcr.io/simzy420/face-swap-runpod-worker:latest`, built by GitHub Actions on push to `main`.
- Endpoint: scale to zero (`workersMin=0`), 48 GB+ GPU, `Qwen/Qwen-Image-2.1` attached as a RunPod cached model.
- Input/output contract: see the docstring at the top of `handler.py`.

Contains no weights and no secrets. Qwen-Image-2.1 and the Viggle turbo LoRA are under the Qwen
Research License (non-commercial); BFS LoRA is MIT.
