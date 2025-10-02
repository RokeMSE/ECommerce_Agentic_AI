from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import torch
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from typing import Dict, List
import json
import boto3
import kubernetes
from kubernetes import client, config

class MultimodalDataset(Dataset):
    """Custom dataset for multimodal review data"""
    
    def __init__(self, data_path: str, processor):
        self.processor = processor
        self.data = self._load_data(data_path)
    
    def _load_data(self, path: str) -> List[Dict]:
        # Load from S3 or local
        with open(path, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as instruction-tuning example
        prompt = f"USER: <image>\nWhat is the sentiment of this product review: '{item['text']}'?\nASSISTANT: {item['sentiment']}"
        
        # Process with LLaVA processor
        inputs = self.processor(
            text=prompt,
            images=item['image_path'],
            return_tensors="pt"
        )
        
        return inputs

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def fetch_labeled_data(s3_bucket: str, min_quality: float = 0.75) -> str:
    """
    Fetch labeled data from S3 that meets quality threshold.
    """
    s3 = boto3.client('s3')
    
    # List all labeled data
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix='labeled/')
    
    high_quality_data = []
    
    for obj in response.get('Contents', []):
        data = s3.get_object(Bucket=s3_bucket, Key=obj['Key'])
        review = json.loads(data['Body'].read())
        
        if review.get('quality_score', 0) >= min_quality:
            high_quality_data.append(review)
    
    # Save consolidated dataset
    output_path = '/tmp/training_data.json'
    with open(output_path, 'w') as f:
        json.dump(high_quality_data, f)
    
    print(f"Fetched {len(high_quality_data)} high-quality reviews")
    return output_path

@task
def train_sentiment_model(
    data_path: str,
    config: Dict
) -> str:
    """
    Fine-tune LLaVA model for sentiment analysis using DeepSpeed.
    """
    # Initialize MLflow tracking
    mlflow.set_experiment("multimodal-sentiment")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config)
        
        # Load model
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf"
        )
        
        # Prepare dataset
        dataset = MultimodalDataset(data_path, processor)
        # Training arguments with DeepSpeed
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=config.get('epochs', 3),
            per_device_train_batch_size=config.get('batch_size', 4),
            gradient_accumulation_steps=4,
            learning_rate=config.get('learning_rate', 2e-5),
            fp16=True,
            deepspeed="./ds_config.json",  # DeepSpeed configuration
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="mlflow"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,  # Should be separate eval set
        )
        
        trainer.train()
        # Save model
        model_path = f"./models/sentiment_vlm_{mlflow.active_run().info.run_id}"
        trainer.save_model(model_path)
        
        # Log model to MLflow
        mlflow.log_artifacts(model_path)
        return model_path

@task
def evaluate_model(model_path: str, test_data_path: str) -> Dict:
    """
    Comprehensive model evaluation with multiple metrics.
    """    
    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    predictions = []
    ground_truth = []
    confidences = []
    model.eval()
    with torch.no_grad():
        for item in test_data:
            prompt = f"USER: <image>\nWhat is the sentiment of this product review: '{item['text']}'?\nASSISTANT:"
            
            inputs = processor(
                text=prompt,
                images=item['image_path'],
                return_tensors="pt"
            )
            
            outputs = model.generate(**inputs, max_new_tokens=10)
            prediction = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Parse prediction
            pred_sentiment = _parse_sentiment(prediction)
            predictions.append(pred_sentiment)
            ground_truth.append(item['sentiment'])
    
    # Calculate metrics
    report = classification_report(
        ground_truth,
        predictions,
        output_dict=True
    )
    
    cm = confusion_matrix(ground_truth, predictions)
    metrics = {
        "accuracy": report['accuracy'],
        "precision_macro": report['macro avg']['precision'],
        "recall_macro": report['macro avg']['recall'],
        "f1_macro": report['macro avg']['f1-score'],
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            cls: {
                "precision": report[cls]['precision'],
                "recall": report[cls]['recall'],
                "f1": report[cls]['f1-score']
            }
            for cls in ['positive', 'negative', 'neutral']
            if cls in report
        }
    }
    
    # Log to MLflow
    mlflow.log_metrics(metrics)
    return metrics

@task
def adversarial_robustness_test(model_path: str) -> Dict:
    """
    Test model robustness against adversarial inputs.
    """
    test_cases = [
        # Typos and misspellings
        {"text": "Thiss produkt is amzing!", "expected": "positive"},
        # Sarcasm
        {"text": "Oh great, another broken product. Just what I needed.", "expected": "negative"},
        # Mixed sentiment
        {"text": "The quality is good but shipping was terrible.", "expected": "neutral"},
        # Low-quality images (blur, noise)
        {"image_augmentation": "gaussian_blur", "expected_degradation": 0.1},
        # Out-of-distribution
        {"text": "This review is about politics, not products.", "expected": "ood_detection"}
    ]
    
    model = LlavaForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    robustness_scores = {
        "typo_handling": 0.0,
        "sarcasm_detection": 0.0,
        "mixed_sentiment": 0.0,
        "image_noise_robustness": 0.0,
        "ood_detection": 0.0
    }
    
    # Test each adversarial case
    for case in test_cases:
        # Implementation would test each case
        pass
    
    return robustness_scores

@task
def deploy_model_to_registry(model_path: str, metrics: Dict) -> str:
    """
    Register model in MLflow Model Registry with metadata.
    """
    # Register model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="multimodal-sentiment-analyzer"
    )
    
    # Add metadata
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(
        name="multimodal-sentiment-analyzer",
        version=registered_model.version,
        key="metrics",
        value=json.dumps(metrics)
    )
    
    # Transition to staging if metrics are good
    if metrics['f1_macro'] > 0.85:
        client.transition_model_version_stage(
            name="multimodal-sentiment-analyzer",
            version=registered_model.version,
            stage="Staging"
        )
    
    return registered_model.version

@flow(name="training-pipeline")
def training_pipeline(config: Dict):
    """
    Main orchestration flow for the entire training pipeline.
    """
    # Fetch and prepare data
    data_path = fetch_labeled_data(
        s3_bucket=config['s3_bucket'],
        min_quality=config.get('min_quality', 0.75)
    )
    
    # Train
    model_path = train_sentiment_model(
        data_path=data_path,
        config=config
    )
    
    # Evaluate on test set
    metrics = evaluate_model(
        model_path=model_path,
        test_data_path=config['test_data_path']
    )
    
    print(f"Evaluation Metrics: {json.dumps(metrics, indent=2)}")
    
    # Robust testing
    robustness = adversarial_robustness_test(model_path)
    print(f"Robustness Scores: {json.dumps(robustness, indent=2)}")
    
    # Deploy to registry
    if metrics['f1_macro'] > config.get('deployment_threshold', 0.80):
        version = deploy_model_to_registry(model_path, metrics)
        print(f"Model deployed to registry: version {version}")  
        # Trigger deployment to inference service
        trigger_inference_deployment(version)
    else:
        print(f"Model performance below threshold. Not deploying.")
    
    return {
        "model_path": model_path,
        "metrics": metrics,
        "robustness": robustness
    }

@task
def trigger_inference_deployment(model_version: str):
    """
    Update inference service to use new model version.
    Uses blue-green deployment strategy.
    """
    # Load Kubernetes config
    config.load_incluster_config()
    
    # Update deployment with new model version
    apps_v1 = client.AppsV1Api()
    deployment = apps_v1.read_namespaced_deployment(
        name="inference-service",
        namespace="production"
    )
    
    # Update container environment variable
    for container in deployment.spec.template.spec.containers:
        if container.name == "inference":
            container.env.append(
                client.V1EnvVar(
                    name="MODEL_VERSION",
                    value=model_version
                )
            )
    
    # Apply update (rolling update)
    apps_v1.patch_namespaced_deployment(
        name="inference-service",
        namespace="production",
        body=deployment
    )
    
    print(f"Rolling update initiated for model version {model_version}")

def _parse_sentiment(text: str) -> str:
    """Parse sentiment from model output"""
    text_lower = text.lower()
    if "positive" in text_lower:
        return "positive"
    elif "negative" in text_lower:
        return "negative"
    else:
        return "neutral"

# DeepSpeed configuration
DS_CONFIG = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,  # ZeRO Stage 2
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 4,
    "wall_clock_breakdown": False
}

# Save DeepSpeed config
with open('ds_config.json', 'w') as f:
    json.dump(DS_CONFIG, f, indent=2)

if __name__ == "__main__":
    # Example configuration
    config = {
        "s3_bucket": "multimodal-reviews-raw",
        "test_data_path": "/data/test_set.json",
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "min_quality": 0.75,
        "deployment_threshold": 0.82
    }
    result = training_pipeline(config)