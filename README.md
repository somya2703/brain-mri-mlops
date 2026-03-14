# Brain MRI Tumor Detection — End-to-End MLOps Pipeline

## Project Overview
This project demonstrates an end-to-end MLOps pipeline for automated brain tumor detection using MRI images. The system trains a deep learning model to classify MRI scans and deploys it through a scalable inference pipeline that supports both real-time predictions via an API and batch inference for large datasets.

It is a machine learning model built in a production-ready ML system that includes:

* Model training
* Experiment tracking
* Model export
* Containerized deployment
* GPU-accelerated inference
* Monitoring

The project represents a complete lifecycle of an ML system, from raw data to production inference.

---
This project is not limited to training a model rather it addresses how such models are deployed and maintained in real world systems.
It is a full MLOps pipeline that takes a trained brain tumor classification model and deploys it as a reproducible, containerized, GPU-accelerated inference service.

---
## System Capabilities

The deployed system supports:

* Real-time MRI classification through an API
* Batch prediction over large datasets
* GPU-accelerated inference
* Experiment tracking for model development
* Containerized reproducible environments
* Monitoring of model performance and system metrics



---
## MLOps Stack Used

This project uses a modern MLOps stack designed to reflect **industry best practices for deploying machine learning systems**.

### Machine Learning Framework
* **PyTorch** – Used for training the brain MRI classification model.

### Model Interoperability
* **ONNX (Open Neural Network Exchange)** – Used to export the trained model into a portable format.

### High-Performance Inference
* **ONNX Runtime** – Provides optimized model inference with support for CPU and GPU execution providers.

### API Serving
* **FastAPI** – A high-performance web framework used to build the real-time inference API.

### Containerization
* **Docker** – Ensures reproducible environments and simplifies deployment across systems.

### GPU Acceleration
* **CUDA-enabled containers** allow the system to leverage GPU hardware for faster inference.

### Experiment Tracking
* **MLflow** – Used for tracking model experiments, parameters, and metrics during training.

### Monitoring
* **Prometheus** – Collects runtime metrics from the inference API to monitor system performance.
* **Grafana** – Visualizes the metrics collected by Prometheus in interactive dashboards for easy monitoring and analysis.
---

*Further Reading: [Results](Results/README.md), [Architecture](architecture_diagrams.md), [No Docker Version](no_docker_version.md).*
