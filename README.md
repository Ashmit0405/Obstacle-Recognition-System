# OBSTACLE-RECOGNITION-SYSTEM
This project implements a multi-model perception system for indoor and outdoor navigation assistance, designed to be robust, low-latency, and safety-oriented.
Instead of relying on a single monolithic model, the system decomposes perception into specialized, cooperating modules, each responsible for a specific aspect of scene understanding and then uses agentic-ai to help assist with the directions.

The architecture prioritizes:
- real-time performance
- robustness to edge cases
- graceful degradation
- modular extensibility

## Setup
```bash
pip install torch torchvision opencv-python einops timm onnxruntime ultralytics fastapi pydantic uvicorn
wget https://huggingface.co/onnx-community/metric3d-vit-large/resolve/main/onnx/model.onnx
pip install -U transformers accelerate
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt
```

## Run
```bash
ngrok config add-authtoken <AUTH-TOKEN>
uvicorn main:app --host 0.0.0.0 --port 8000
ngrok http 8000
```