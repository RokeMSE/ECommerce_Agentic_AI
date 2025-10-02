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