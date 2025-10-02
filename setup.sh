#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "=================================================="
echo "  E-Commerce Agentic AI System - Setup Script"
echo "=================================================="
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
# API Keys (REQUIRED - Update these!)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here

# AWS/MinIO Configuration
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_REGION=us-east-1
S3_ENDPOINT_URL=http://minio:9000
S3_BUCKET_NAME=multimodal-reviews-raw

# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mlflow

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=development
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
  - name: inference-service
    url: http://inference-service:8000
    protocol: http
    connect_timeout: 60000
    write_timeout: 60000
    read_timeout: 60000

routes:
  - name: analyze-route
    service: inference-service
    paths:
      - /api/analyze
    methods:
      - POST
    strip_path: false
  
  - name: health-route
    service: inference-service
    paths:
      - /api/health
    methods:
      - GET
    strip_path: false

plugins:
  - name: cors
    config:
      origins:
        - "*"
      methods:
        - GET
        - POST
        - PUT
        - DELETE
        - OPTIONS
      headers:
        - Accept
        - Authorization
        - Content-Type
      credentials: true
      max_age: 3600
EOF
fi

# Create database initialization script
if [ ! -f scripts/init-db.sql ]; then
    print_status "Creating database initialization script..."
    cat > scripts/init-db.sql << 'EOF'
-- Create databases
CREATE DATABASE IF NOT EXISTS mlflow;
CREATE DATABASE IF NOT EXISTS prefect;

-- Connect to app database
\c postgres;

-- Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    id SERIAL PRIMARY KEY,
    review_id VARCHAR(255) UNIQUE NOT NULL,
    text TEXT NOT NULL,
    sentiment VARCHAR(50),
    confidence FLOAT,
    product_name VARCHAR(255),
    rating FLOAT,
    is_synthetic BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO reviews (review_id, text, sentiment, confidence, product_name, rating, is_synthetic)
VALUES 
    ('demo-001', 'This product exceeded my expectations! Amazing quality.', 'positive', 0.95, 'Demo Product A', 5.0, false),
    ('demo-002', 'Terrible experience. Product broke after one day.', 'negative', 0.92, 'Demo Product B', 1.0, false),
    ('demo-003', 'It''s okay. Not great, not terrible.', 'neutral', 0.78, 'Demo Product C', 3.0, false)
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