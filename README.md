# E-Commerce Agentic Multimodal Sentiment Analysis System
## About
An end-to-end, production-ready agentic AI system for multimodal sentiment analysis. This system autonomously synthesizes review text and images, uses this data to train advanced Vision-Language Models, and provides a robust API for sentiment analysis and hybrid retrieval, effectively solving the "cold start" problem for niche e-commerce categories.

## Repository Structure
```
ecommerce_agentic_ai/
├── agents/                  # Autonomous agents for scraping and synthesis
│   ├── scraping_agent/      # Agent used for scraping reviews data from web
│   └── synthetic_agent/     # Agent used for synthesizing review data
├── backend/                 # Backend microservices
│   ├── inference_service/   # FastAPI app for model inference
│   └── training_service/    # Manages model training and evaluation
├── frontend/                # Gradio user interface
├── kong/                    # Kong API Gateway configuration
├── scripts/           # Initialization scripts (like init_db for DB)
├── models/            # Directory for storing model artifacts (mounted)
├── data/              # Directory for local data storage (mounted to MinIO)
├── docker-compose.yml # Main Docker Compose file for orchestration
└── setup.sh           # Interactive setup script
```

## Overview
1. **Autonomous Data Acquisition**: An intelligent scraping agent uses an LLM to decide which web sources to target and dynamically discovers how to extract multimodal review data (text and images).
2. **Advanced Data Synthesis**: A synthesis agent generates high-fidelity, domain-specific text and images, creating large, balanced datasets for training. The quality is ensured through an adversarial validation process.
3. **Multimodal Understanding**: The system fine-tunes a Vision-Language Model (LLaVA) to jointly understand text and images, capturing nuanced sentiment that unimodal models would miss.
4. **Hybrid Retrieval**: A hybrid retrieval system combines dense (semantic) and sparse (keyword) search with a cross-encoder reranker to find the most relevant similar reviews.

## Key Features
- **LLM-Guided Scraping Strategy**: The scraping agent learns from past successes and failures to dynamically decide where to find the highest quality data.
- **Controllable Synthetic Data**: Generate text and images with specific sentiment signals (e.g., a negative review paired with an image of a damaged product).
- **Adversarial Validation**: A discriminator network ensures that synthetic data is indistinguishable from real data, guaranteeing high quality for training.
- **Fine-Tuned Vision-Language Model**: Utilizes a fine-tuned LLaVA v1.6 model for superior contextual understanding of text and images together.
- **Reciprocal Rank Fusion (RRF)**: Intelligently combines semantic search results from Qdrant (vectors) and keyword search results from Elasticsearch (BM25) for robust retrieval.
- **End-to-End MLOps Automation**: Features an automated training pipeline with Prefect, experiment tracking with MLflow, and model versioning/deployment.
- **Microservices Architecture**: Built with Docker for scalability, maintainability, and easy local setup.

## System Architecture
The system is a network of containerized microservices orchestrated by Docker Compose. The architechture design separates concerns, allowing each component to be developed, deployed, and scaled independently.

### Core Services
- **API Gateway (Kong)**: The single entry point for all API requests, handling routing, and security.
- **Frontend (Gradio)**: A simple web UI for interacting with the sentiment analysis model.
- **Inference Service**: A FastAPI backend that serves the sentiment analysis model and orchestrates the hybrid retrieval logic.
- **Agent Services**: Separate containers for the `scraping-agent` and `synthetic-agent`.
- **Training Service**: Manages the model training and evaluation lifecycle.
- **Data Stores**:
    + **PostgreSQL**: Stores structured data for MLflow and other application needs.
    + **Qdrant**: A vector database for fast, efficient semantic search.
    + **Elasticsearch**: Powers keyword-based sparse search.
    + **Redis**: Caches frequent API responses to reduce latency.
    + **MinIO**: An S3-compatible object store for ML artifacts and raw data.

## Technology Stack
| Component            | Technology                                           |
| -------------------- | ---------------------------------------------------- |
| **AI/ML Frameworks** | PyTorch, Transformers, Diffusers                     |
| **Backend Services** | FastAPI, Uvicorn                                     |
| **Frontend** | Gradio                                               |
| **Databases** | PostgreSQL, Qdrant, Elasticsearch, Redis           |
| **Core Models** | LLaVA-v1.6, CLIP, BERT, SDXL                         |
| **Web Scraping** | Playwright, Scrapy                                   |
| **MLOps** | MLflow, Prefect, MinIO                               |
| **Orchestration** | Docker, Docker Compose                               |
| **API Gateway** | Kong                                                 |

## Getting Started

### Prerequisites
  - **Docker & Docker Compose**: Ensure Docker Desktop is installed and running.
  - **NVIDIA GPU**: Recommended for training and inference services for optimal performance.
  - **Git**: For cloning the repository.

### Local Development Setup
1. **Clone the Repository**
    ```bash
    git clone https://github.com/rokemse/ecommerce_agentic_ai.git
    cd ecommerce_agentic_ai
    ```

2. **Run the Setup Script**
This script will create the necessary configuration files and a `.env` file for your secrets.
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

3. **Configure Environment Variables**
The setup script will create a `.env` file. You **must** open it and add your `OPENAI_API_KEY`.
    ```bash
    nano .env
    ```

4. **Start the Services**
The setup script will prompt you to choose a deployment option. For a first run, the "Minimal Demo" is recommended. The services will start automatically.
To start them manually later:
    ```bash
    docker-compose up --build -d
    ```

5. **Access the System**
Once the containers are running, you can access the various components:
    - **Frontend UI**: `http://localhost:7860`
    - **API Gateway**: `http://localhost:8000`
    - **Inference API Docs**: `http://localhost:8080/docs`
    - **MLflow UI**: `http://localhost:5000`
    - **MinIO Console**: `http://localhost:9001` (Login with `minioadmin`/`minioadmin`)

    *Note: The first time you run the `inference-service`, it may take a few minutes to download the CLIP model.*

6.  **Stopping the System**
    ```bash
    docker-compose down
    ```

## Deployment Options
### Option A: Demo
- Services: 8 essential services only.
- Includes:
    - API Gateway (Kong)
    - Inference Service
    - Frontend (Gradio)
    - Databases (PostgreSQL, Redis, Qdrant, Elasticsearch)
    - Storage (MinIO)
- Excludes:
    - Training Service (not needed for inference)
    - Agents (not needed for basic demo)
    - Monitoring (optional for demo)

### Option B: Full System
- Services: All 13 services.
- Additional services:
    + Scraping Agent
    + Synthetic Agent
    + Training Service
    + MLflow
    + Prometheus
    + Grafana
    + Prefect

## Demo run:
This is explicitly designed as a placeholder for the full, fine-tuned LLaVA model, which is too resource-intensive for a standard local deployment. 

Here is how the classification works, as detailed in `backend/inference_service/main.py`: 
- Predefined Keyword Lists: The system has hardcoded lists of positive and negative words.
    + `Negative words: ['terrible', 'awful', 'bad', 'worst', 'hate', 'horrible', 'poor', 'disappointed', 'waste', 'broken']`
    + `Positive words: ['excellent', 'amazing', 'great', 'best', 'love', 'wonderful', 'perfect', 'fantastic', 'awesome', 'exceeded']`
- Keyword Counting: When a review text is submitted, the code converts the text to lowercase and counts how many words from the positive and negative lists
appear in it.
- Classification Logic: The final sentiment is determined by a simple comparison of the counts:
    + If the count of positive words is greater than the count of negative words, the sentiment is classified as "positive".
    + If the count of negative words is greater, it's classified as "negative".
    + If the counts are equal (or both are zero), the sentiment is "neutral".

## What's Not Implemented (Limitations)
### Full ML Pipeline (Placeholder)
- Status: Pseudo-code + simplified

1. **LLaVA Model**: Current uses rule-based classifier
- Reason: 13GB model, requires GPU, long download
- Production: Needs fine-tuned model artifacts

2. **Training Pipeline**: Structure exists, needs real data
- Current: Prefect flows defined
- Missing: Actual training execution, model artifacts

3. **Synthetic Agent**: Code structure only
- Reason: LLaMA 3.1 + SDXL = 30GB+, requires **minimum** 24GB VRAM
- Production: Needs really powerful GPU

### Retrieval System (Not Integrated)
- Status: Code exists, not connected
- Hybrid Retrieval: Implementation in `hybrid_dense_parse_retrieval_approach.py`
- Missing: Integration with inference endpoint
- Reason: Demo focuses on sentiment analysis first

### Automated Labeling (Suggestion Only)
- Status: Documentation only
- Reason: Requires paid API calls, demo uses existing label

## Running the full Agentic Pipelines (AFTER ALL IMPLEMENTATION)
After the **FULL** infrastructure is running, you can execute the core agentic workflows. These commands should be run from within the respective service containers or configured to run as standalone scripts that connect to the Docker network.

### 1. Data Ingestion
To start the autonomous scraping process, run the scraping agent. This agent will use its LLM-guided strategy to find and extract reviews.

```bash
# Execute inside the running scraping_agent container
docker-compose exec scraping-agent python openai_webscraping_agent.py
```

### 2. Data Synthesis
Once you have some initial data, run the synthesis agent to create a larger, balanced dataset.

```bash
# Execute inside the running synthetic_agent container
docker-compose exec synthetic-agent python controllable_synthetic_agent.py
```

### 3. Model Training
With a complete dataset, you can then execute the automated training pipeline.

```bash
# Execute inside the running training_service container
docker-compose exec training-service python distributed_training.py
```

> **This will fine-tune the LLaVA model, evaluate it, and register the new version in MLflow if it meets the performance criteria.**

## Contributing
Contributions are welcome\! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or improvements.