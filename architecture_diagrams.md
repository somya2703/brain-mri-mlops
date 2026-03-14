# Brain MRI Tumor Detection — End-to-End MLOps Pipeline

## architecture diagrams


## Project Structure
```
brain-mri-mlops/
│
├── api/
│ └── main.py
│
├── inference/
│ ├── batch_inference.py
│ └── tensorrt_engine.py
│
├── src/
│ ├── dataset.py
│ ├── export_onnx.py
│ ├── model.py
│ └── train.py
│
├── visualization/
│ ├── visualize_predictions.py
│
├── docker/
│ ├── Dockerfile.api
│ ├── Dockerfile.batch
│ └── Dockerfile.training
│
├── monitoring/
│ └── prometheus.yml
│
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── gpu_test_script.py
├── debug_model.py
├── no_docker_run.py
│
└── .gitignore
```
