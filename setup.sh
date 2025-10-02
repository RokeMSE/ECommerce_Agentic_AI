#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "  E-Commerce Agentic AI System - Setup Script"
echo -e "${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker Desktop first."
    echo "Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi
print_status "Docker is installed"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install it first."
    exit 1
fi
print_status "Docker Compose is installed"

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker Desktop."
    exit 1
fi
print_status "Docker is running"

# Create required directories
print_status "Creating required directories..."
mkdir -p models data monitoring/grafana-dashboards scripts kong

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating from template..."
    cat > .env << 'EOF'
# OpenAI API Key (ADD YOURS)
OPENAI_API_KEY=""

# AWS/MinIO Configuration
AWS_ACCESS_KEY_ID=minioadmin # Will be replaced in production with real AWS credentials
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_REGION=""
S3_ENDPOINT_URL=http://minio:9000
S3_BUCKET_NAME=multimodal-reviews-raw

# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mlflow
DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}"

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# Qdrant Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Elasticsearch Configuration
ELASTICSEARCH_URL=http://elasticsearch:9200
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=changeme

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=postgresql://postgres:postgres@postgres:5432/mlflow
MLFLOW_ARTIFACT_ROOT=s3://mlflow-artifacts/

# Model Configuration
MODEL_VERSION=v1.0
MODEL_PATH=/models/sentiment_vlm

# API Configuration
API_URL=http://api-gateway:8000
INFERENCE_SERVICE_URL=http://inference-service:8000

# Prefect Configuration
PREFECT_API_URL=http://prefect-server:4200/api

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
GRAFANA_ADMIN_PASSWORD=admin

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=development
MAX_WORKERS=4 # Number of parallel workers for inference
EOF
    print_warning "Please edit .env file and add your API keys!"
    print_warning "Run: nano .env (or use your preferred editor)"
    
    read -p "Press Enter after updating .env file (or Ctrl+C to exit)..."
fi
print_status ".env file exists"

# Create monitoring configuration
if [ ! -f monitoring/prometheus.yml ]; then
    print_status "Creating Prometheus configuration..."
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'inference-service'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['inference-service:8000']
EOF
fi

if [ ! -f monitoring/grafana-datasources.yml ]; then
    print_status "Creating Grafana datasource configuration..."
    cat > monitoring/grafana-datasources.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
fi

# Create Kong configuration
if [ ! -f kong/kong.yml ]; then
    print_status "Creating Kong API Gateway configuration..."
    cat > kong/kong.yml << 'EOF'
_format_version: "3.0"

services:
  - name: inference-api-service
    url: http://inference-service:8000 # Pointing to the inference service defined in docker-compose.yml
    routes:
      - name: inference-routes
        paths:
          - /api
        strip_path: true  # Remove /api from the path before forwarding to the service
EOF
fi

# Create database initialization script
if [ ! -f scripts/init-db.sql ]; then
    print_status "Creating database initialization script..."
    cat > scripts/init-db.sql << 'EOF'
-- Create databases for different services
CREATE DATABASE IF NOT EXISTS mlflow;
CREATE DATABASE IF NOT EXISTS kong;
CREATE DATABASE IF NOT EXISTS prefect;
CREATE DATABASE IF NOT EXISTS app;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;
GRANT ALL PRIVILEGES ON DATABASE kong TO postgres;
GRANT ALL PRIVILEGES ON DATABASE prefect TO postgres;
GRANT ALL PRIVILEGES ON DATABASE app TO postgres;

-- Connect to app database and create tables
\c app;

-- Reviews metadata table
CREATE TABLE IF NOT EXISTS reviews (
    id SERIAL PRIMARY KEY,
    review_id VARCHAR(255) UNIQUE NOT NULL,
    text TEXT NOT NULL,
    sentiment VARCHAR(50),
    confidence FLOAT,
    product_name VARCHAR(255),
    rating FLOAT,
    source_url TEXT,
    image_urls TEXT[],
    is_synthetic BOOLEAN DEFAULT FALSE,
    quality_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scraping jobs table
CREATE TABLE IF NOT EXISTS scraping_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    target_url TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    reviews_collected INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training jobs table
CREATE TABLE IF NOT EXISTS training_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    model_version VARCHAR(50),
    status VARCHAR(50) DEFAULT 'pending',
    metrics JSONB,
    config JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model registry table
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_path TEXT NOT NULL,
    metrics JSONB,
    stage VARCHAR(50) DEFAULT 'none',
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, version)
);

-- Create indexes for better performance
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment);
CREATE INDEX idx_reviews_created_at ON reviews(created_at);
CREATE INDEX idx_scraping_jobs_status ON scraping_jobs(status);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_model_registry_stage ON model_registry(stage);

-- Insert sample data for testing (optional)
INSERT INTO reviews (review_id, text, sentiment, confidence, product_name, rating, is_synthetic)
VALUES 
    ('demo-001', 'This product exceeded my expectations! The quality is amazing and it works perfectly.', 'positive', 0.95, 'Demo Product A', 5.0, false),
    ('demo-002', 'Terrible experience. The product broke after one day of use. Very disappointed.', 'negative', 0.92, 'Demo Product B', 1.0, false),
    ('demo-003', 'It''s okay. Not great, not terrible. Does what it''s supposed to do.', 'neutral', 0.78, 'Demo Product C', 3.0, false)
ON CONFLICT (review_id) DO NOTHING;
EOF
fi

print_status "All configuration files created"

# Ask user which deployment option
echo ""
echo "Choose deployment option:"
echo "1) Minimal Demo (Inference + Frontend only) - Recommended for first run"
echo "2) Full System (All services including agents)"
echo ""
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        print_status "Starting minimal demo deployment..."
        docker-compose up -d postgres redis minio minio-init qdrant elasticsearch api-gateway inference-service frontend
        ;;
    2)
        print_status "Starting full system deployment..."
        docker-compose up -d
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
print_status "Waiting for services to start (30 seconds)..."
sleep 30

# Check service health
echo ""
echo "Checking service health..."
echo ""

# Check inference service
if curl -s http://localhost:8080/health | grep -q "healthy"; then
    print_status "Inference Service: Running"
else
    print_warning "Inference Service: Not ready yet (may still be loading models)"
fi

# Check frontend
if curl -s http://localhost:7860 > /dev/null 2>&1; then
    print_status "Frontend: Running"
else
    print_warning "Frontend: Not ready yet"
fi

# Check API Gateway
if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    print_status "API Gateway: Running"
else
    print_warning "API Gateway: Not ready yet"
fi

echo ""
echo -e "${GREEN}=================================================="
echo "  Setup Complete!"
echo -e "==================================================${NC}"
echo ""
echo "Access your services at:"
echo ""
echo "  Frontend (Gradio):     http://localhost:7860"
echo "  API Documentation:     http://localhost:8080/docs"
echo "  API Gateway:           http://localhost:8000"
echo "  MLflow UI:             http://localhost:5000"
echo "  MinIO Console:         http://localhost:9001 (minioadmin/minioadmin)"
echo ""
echo "Quick Test:"
echo '  curl -X POST "http://localhost:8000/api/analyze" -F "text=Great product!"'
echo ""
echo "View logs:"
echo "  docker-compose logs -f inference-service"
echo ""
echo "Stop services:"
echo "  docker-compose down"
echo ""
print_warning "Note: First run may take 2-3 minutes to download CLIP model"
echo ""