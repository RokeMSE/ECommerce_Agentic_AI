# E-Commerce Agentic AI - API Guide
## Introduction
The API allows you to analyze the sentiment of product reviews by providing text and an optional image.
All API requests are routed through the **Kong API Gateway**, which acts as the single entry point to the system.
**Base URL**: `http://localhost:8000`

## Authentication
`NaN`

In a production environment, the API Gateway (Kong) would be configured to handle authentication (API keys, OAuth2). For this local development setup, authentication is disabled.

## Endpoints
### 1. Analyze Review Sentiment
Performs a multimodal sentiment analysis on the provided text and optional image, returning a sentiment classification, confidence score, and other metadata.

`POST /api/analyze`

#### Request Body
The endpoint expects a `multipart/form-data` request, **not a JSON payload**.
| Parameter | Type   | Required | Description                                     |
| --------- | ------ | -------- | ----------------------------------------------- |
| `text`    | string | **Yes** | The text content of the product review.         |
| `image`   | file   | No       | An optional image file associated with the review. |

#### Example Request (`curl`)
Here is an example of how to call the API using `curl` from your terminal.

**Text-only analysis:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "text=This product is absolutely amazing, exceeded all my expectations!"
```

**Multimodal (text + image) analysis:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "text=The product arrived broken, completely unusable." \
  -F "image=@/path/to/your/image.jpg"
```

#### Success Response (200 OK)
The API returns a JSON object with the analysis results.

```json
{
  "sentiment": "positive",
  "confidence": 0.85,
  "modality": "multimodal",
  "similar_reviews": [],
  "processing_time_ms": 150.75,
  "cached": false
}
```

**Response Fields:**
| Field                | Type    | Description                                                                                                                              |
| -------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `sentiment`          | string  | The predicted sentiment. Can be `positive`, `negative`, or `neutral`.                                                 |
| `confidence`         | float   | A score between 0.0 and 1.0 indicating the model's confidence in its prediction.                                           |
| `modality`           | string  | Indicates the type of input used for analysis: `text_only` or `multimodal`.                                                |
| `similar_reviews`    | array   | *[Future Implementation]* An array of similar reviews retrieved from the vector database. Currently returns an empty array.              |
| `processing_time_ms` | float   | The total time taken to process the request, in milliseconds.                                                        |
| `cached`             | boolean | `true` if the result was served from the Redis cache, `false` otherwise.                                                    |

#### Error Responses
| Status Code | Error Code          | Description                                                                 |
| ----------- | ------------------- | --------------------------------------------------------------------------- |
| `400`       | Bad Request         | The `text` field was missing or empty.                 |
| `500`       | Internal Server Error | An unexpected error occurred during analysis or embedding generation.      |
| `503`       | Service Unavailable | The service has started, but the machine learning models failed to load. |

### 2. Health Check
This endpoint provides a simple health check to verify that the inference service is running and its dependencies (like the cache and models) are available.

`GET /api/health`

#### Example Request (`curl`)
```bash
curl -X GET "http://localhost:8000/api/health"
```

#### Success Response (200 OK)
```json
{
  "status": "healthy",
  "device": "cuda",
  "cache_connected": true,
  "models_loaded": true
}
```

**Response Fields:**
| Field             | Type    | Description                                                                     |
| ----------------- | ------- | ------------------------------------------------------------------------------- |
| `status`          | string  | `healthy` if the service and models are loaded, `unhealthy` otherwise.         |
| `device`          | string  | The compute device being used for model inference (either `cuda` or `cpu`).       |
| `cache_connected` | boolean | `true` if the service has a successful connection to the Redis cache.             |
| `models_loaded`   | boolean | `true` if the required machine learning models have been loaded into memory.      |