# An E-Commerce Agentic Multimodal Sentiment Analysis System
An agentic AI system that synthesizes review text and images from public sources, and uses these data sources to train ML or LLM models for sentiment analysis.

An end-to-end, production-ready agentic AI system for multimodal sentiment analysis with autonomous data synthesis, hybrid retrieval, and continuous learning capabilities.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Detailed Documentation](#detailed-documentation)
- [Evaluation Results](#evaluation-results)
- [Contributing](#contributing)

## Overview
This system addresses the "cold start" problem in sentiment analysis for niche e-commerce categories by implementing:

1. **Autonomous Data Acquisition**: Intelligent web scraping agents that learn which sources provide high-quality data
2. **Advanced Data Synthesis**: LLM and diffusion-based generation of realistic, domain-specific reviews and product images
3. **Multimodal Understanding**: Fine-tuned Vision-Language Models (LLaVA) for joint text-image sentiment analysis
4. **Hybrid Retrieval**: Combination of dense (CLIP) and sparse (BM25) retrieval with cross-encoder reranking
5. **Self-Improving Pipeline**: Continuous learning with adversarial validation and quality monitoring

## Architecture



## Key Features

### 1. Intelligent Data Acquisition
- **LLM-Guided Strategy**: Claude Sonnet decides which sources to scrape based on historical success rates
- **Adaptive Selectors**: Automatically discovers CSS selectors for review extraction
- **Multi-Protocol Support**: Scrapy for static sites, Playwright for JavaScript-heavy applications
- **Deduplication**: Content-based hashing prevents duplicate reviews

### 2. Advanced Data Synthesis
- **Style-Transfer Text Generation**: Fine-tuned Llama 3.1 mimics domain-specific writing patterns
- **Controllable Image Generation**: SDXL + ControlNet creates product images with sentiment-specific visual cues
- **Adversarial Validation**: Discriminator network ensures synthetic data is indistinguishable from real data
- **Quality Filtering**: Multi-stage validation with LLM-based quality assessment

### 3. Multimodal Sentiment Analysis
- **Vision-Language Model**: Fine-tuned LLaVA-v1.6 for joint text-image understanding
- **Contextual Analysis**: Captures nuanced sentiment from text-image interactions
- **Confidence Calibration**: Provides well-calibrated confidence scores

### 4. Hybrid Retrieval System
- **Dense Retrieval**: CLIP embeddings for semantic similarity
- **Sparse Retrieval**: Elasticsearch BM25 for keyword matching
- **Reciprocal Rank Fusion**: Combines dense and sparse results
- **Cross-Encoder Reranking**: Final precision boost with ms-marco reranker
- **Multi-Vector Search**: Separate text/image embeddings for fine-grained matching

### 5. Production-Ready Infrastructure
- **Microservices Architecture**: Independent, scalable services
- **Comprehensive Monitoring**: Prometheus + Grafana dashboards
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | PyTorch, Transformers, Diffusers |
| **Backend** | FastAPI, Prefect |
| **Frontend** | Gradio, React (TypeScript) |
| **Databases** | PostgreSQL+pgvector, Qdrant, Elasticsearch, Redis |
| **ML/LLM** | LLaVA-v1.6, Llama 3.1, SDXL, CLIP |
| **Orchestration** | Kubernetes (EKS), Docker |
| **Cloud** | AWS (S3, SageMaker, EKS, RDS) |
| **Monitoring** | Prometheus, Grafana, MLflow |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (for training and inference)
- Python 3.10+
- AWS Account (for cloud deployment)

### Local Development

```bash
# Clone repository
git clone https://github.com/your-username/multimodal-sentiment-system.git
cd multimodal-sentiment-system

# Set environment variables
cp .env.example .env
# Edit .env with your API keys (Claude, HuggingFace, AWS)

# Start all services
docker-compose up -d

# Access services
# - Frontend: http://localhost:7860
# - API Gateway: http://localhost:8000
# - MLflow: http://localhost:5000
# - Grafana: http://localhost:3000
```

### Run Training Pipeline

```bash
# Activate Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data ingestion
python agents/ingestion_agent/run.py --config configs/scraping_config.yaml

# Run data synthesis
python agents/synthesis_agent/run.py --target-size 10000

# Train sentiment model
python backend/training_service/train.py --config configs/training_config.yaml
```

## 📁 Repository Structure

```
multimodal-sentiment-system/
├── backend/
│   ├── inference_service/
│   │   ├── main.py              # FastAPI application
│   │   ├── models.py            # Pydantic models
│   │   ├── model_manager.py     # Model loading and inference
│   │   └── Dockerfile
│   ├── training_service/
│   │   ├── train.py             # Training orchestration
│   │   ├── dataset.py           # Custom PyTorch datasets
│   │   ├── evaluation.py        # Evaluation metrics
│   │   └── Dockerfile
│   └── api_gateway/
│       └── kong.yml             # Kong configuration
├── agents/
│   ├── ingestion_agent/
│   │   ├── scraper.py           # Intelligent web scraping
│   │   ├── strategy_memory.py   # Learning from past scrapes
│   │   └── Dockerfile
│   ├── synthesis_agent/
│   │   ├── text_synthesizer.py  # LLM-based text generation
│   │   ├── image_synthesizer.py # Diffusion-based image generation
│   │   ├── validator.py         # Adversarial validation
│   │   └── Dockerfile
│   ├── labeling_agent/
│   │   ├── zero_shot_labeler.py # LLM-based labeling
│   │   ├── active_learning.py   # Uncertainty sampling
│   │   └── Dockerfile
│   └── control_agent/
│       ├── orchestrator.py      # Prefect workflows
│       └── Dockerfile
├── frontend/
│   ├── gradio_app/
│   │   ├── app.py               # Gradio interface
│   │   └── Dockerfile
│   └── react_app/               # Optional React frontend
│       ├── src/
│       └── Dockerfile
├── infra/
│   ├── docker-compose.yml       # Local development
│   ├── k8s/                     # Kubernetes manifests
│   │   ├── deployments/
│   │   ├── services/
│   │   ├── configmaps/
│   │   └── ingress.yaml
│   ├── terraform/               # Infrastructure as Code
│   └── monitoring/
│       ├── prometheus.yml
│       └── grafana/
├── scripts/
│   ├── init_database.py
│   ├── seed_data.py
│   └── deploy.sh
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_evaluation_analysis.ipynb
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── ARCHITECTURE.md          # Detailed architecture
│   ├── API.md                   # API documentation
│   ├── DEPLOYMENT.md            # Deployment guide
│   └── EVALUATION.md            # Evaluation methodology
├── configs/
│   ├── scraping_config.yaml
│   ├── synthesis_config.yaml
│   └── training_config.yaml
├── .github/
│   └── workflows/
│       ├── ci.yml               # Continuous Integration
│       └── cd.yml               # Continuous Deployment
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## 📚 Detailed Documentation

### Architecture & Design
- [System Architecture](docs/ARCHITECTURE.md) - Comprehensive system design
- [Microservices Overview](docs/MICROSERVICES.md) - Detailed service descriptions
- [Data Flow](docs/DATA_FLOW.md) - End-to-end data pipeline

### API Documentation
- [REST API Reference](docs/API.md) - Complete API documentation
- [WebSocket Events](docs/WEBSOCKETS.md) - Real-time event streaming

### Deployment Guides
- [Local Development](docs/LOCAL_SETUP.md) - Development environment setup
- [Kubernetes Deployment](docs/DEPLOYMENT.md) - Production deployment on K8s
- [AWS Setup](docs/AWS_SETUP.md) - AWS infrastructure configuration

### Model Training
- [Training Pipeline](docs/TRAINING.md) - Model training workflow
- [Hyperparameter Tuning](docs/HYPERPARAMETERS.md) - Optimization strategies
- [Model Registry](docs/MODEL_REGISTRY.md) - Version management

### Evaluation
- [Evaluation Framework](docs/EVALUATION.md) -