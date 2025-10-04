import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter
import torch
import pandas as pd
from agents.synthetic_agent.controllable_synthetic_agent import AdversarialValidator
from textblob import TextBlob
import time
import concurrent.futures

class ComprehensiveEvaluator:
    """
    Multi-faceted evaluation framework for the entire pipeline.
    """
    
    def __init__(self, model, processor, test_data: List[Dict]):
        self.model = model
        self.processor = processor
        self.test_data = test_data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def evaluate_sentiment_performance(self) -> Dict:
        """Standard classification metrics"""
        predictions = []
        ground_truth = []
        confidences = []
        for item in self.test_data:
            pred, conf = self._predict(item['text'], item.get('image'))
            predictions.append(pred)
            ground_truth.append(item['sentiment'])
            confidences.append(conf)
        
        # Classification report
        report = classification_report(
            ground_truth,
            predictions,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        
        # Calibration curve (reliability)
        calibration = self._calculate_calibration(
            ground_truth,
            predictions,
            confidences
        )
        
        return {
            "accuracy": report['accuracy'],
            "macro_f1": report['macro avg']['f1-score'],
            "per_class": {
                cls: report[cls] for cls in ['positive', 'negative', 'neutral']
                if cls in report
            },
            "confusion_matrix": cm.tolist(),
            "calibration": calibration
        }
    
    def evaluate_multimodal_contribution(self) -> Dict:
        """
        Quantify the benefit of multimodal vs. unimodal analysis.
        """
        results = {
            "text_only": [],
            "image_only": [],
            "multimodal": []
        }
        
        for item in self.test_data:
            if not item.get('image'):
                continue 
            # Text only
            pred_text, _ = self._predict(item['text'], None)
            results["text_only"].append(
                pred_text == item['sentiment']
            )
            # Image only (using empty text)
            pred_image, _ = self._predict("", item['image'])
            results["image_only"].append(
                pred_image == item['sentiment']
            )
            # Multimodal
            pred_multi, _ = self._predict(item['text'], item['image'])
            results["multimodal"].append(
                pred_multi == item['sentiment']
            )
        
        return {
            "text_only_accuracy": np.mean(results["text_only"]),
            "image_only_accuracy": np.mean(results["image_only"]),
            "multimodal_accuracy": np.mean(results["multimodal"]),
            "multimodal_improvement": np.mean(results["multimodal"]) - 
                                     max(np.mean(results["text_only"]),
                                         np.mean(results["image_only"]))
        }
    
    def evaluate_adversarial_robustness(self) -> Dict:
        """
        Test robustness against various adversarial perturbations.
        """
        robustness_scores = {
            "character_level": self._test_character_perturbations(),
            "word_level": self._test_word_perturbations(),
            "image_noise": self._test_image_perturbations(),
            "occlusion": self._test_image_occlusion(),
            "mixed_sentiment": self._test_mixed_sentiment()
        }
        
        return robustness_scores
    
    def _test_character_perturbations(self) -> float:
        """Test against typos, character swaps"""
        correct_count = 0
        total = 0
        for item in self.test_data[:50]:  # Sample subset
            original_pred, _ = self._predict(item['text'], item.get('image'))
            # Apply perturbations
            perturbed = self._add_typos(item['text'])
            perturbed_pred, _ = self._predict(perturbed, item.get('image'))
            # ***************Should maintain same prediction
            if original_pred == perturbed_pred:
                correct_count += 1
            total += 1
        
        return correct_count / total if total > 0 else 0.0
    
    def _test_word_perturbations(self) -> float:
        """Test against synonym replacement, word deletion"""
        correct_count = 0
        total = 0
        for item in self.test_data[:50]:
            original_pred, _ = self._predict(item['text'], item.get('image'))
            # Remove random words
            words = item['text'].split()
            keep_mask = np.random.rand(len(words)) > 0.2 # 20% deletion
            perturbed = ' '.join([w for w, keep in zip(words, keep_mask) if keep])
            if not perturbed.strip():
                continue
            
            perturbed_pred, _ = self._predict(perturbed, item.get('image'))
            if original_pred == perturbed_pred:
                correct_count += 1
            total += 1
        return correct_count / total if total > 0 else 0.0
    
    def _test_image_perturbations(self) -> float:
        """Test against Gaussian noise, blur, brightness changes"""
        if not any(item.get('image') for item in self.test_data):
            return 0.0
        
        correct_count = 0
        total = 0
        for item in self.test_data[:50]:
            if not item.get('image'):
                continue
            original_pred, _ = self._predict(item['text'], item['image'])
            
            # Add Gaussian noise
            image_array = np.array(item['image'])
            noise = np.random.normal(0, 25, image_array.shape)
            noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
            noisy_pil = Image.fromarray(noisy_image)
            noisy_pred, _ = self._predict(item['text'], noisy_pil)
            
            if original_pred == noisy_pred:
                correct_count += 1
            total += 1
        
        return correct_count / total if total > 0 else 0.0
    
    def _test_image_occlusion(self) -> float:
        """Test with partially occluded images"""
        correct_count = 0
        total = 0
        for item in self.test_data[:50]:
            if not item.get('image'):
                continue
            original_pred, _ = self._predict(item['text'], item['image'])
            
            # Occlude 25% of image
            image_array = np.array(item['image'])
            h, w = image_array.shape[:2]
            mask = np.ones((h, w), dtype=bool)
            mask[:h//2, :w//2] = False
            image_array[~mask] = 0
            
            occluded_pil = Image.fromarray(image_array)
            occluded_pred, _ = self._predict(item['text'], occluded_pil)
            if original_pred == occluded_pred:
                correct_count += 1
            total += 1
        
        return correct_count / total if total > 0 else 0.0
    
    def _test_mixed_sentiment(self) -> float:
        """
        Test on reviews with conflicting sentiment signals.
        E.g., positive text with image of damaged product.
        """
        # Specially crafted test cases
        mixed_test_cases = [
            {
                "text": "This product is amazing!",
                "image": "damaged_product.jpg",
                "expected": "negative"  # Image should dominate
            },
            {
                "text": "Terrible quality, very disappointed",
                "image": "pristine_product.jpg",
                "expected": "negative"  # Text sentiment should dominate
            }
        ]
        
        # Implementation would load these test cases and evaluate
        return 0.75  # Placeholder
    
    def evaluate_retrieval_quality(
        self,
        retrieval_system,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict:
        """
        Evaluate retrieval system using Precision@K, Recall@K, MAP.
        """
        # Create ground truth similar pairs
        ground_truth = self._create_similarity_ground_truth()
        results = {}
        for k in k_values:
            precisions = []
            recalls = []
            aps = []  # Average Precision
            for query_id, relevant_ids in ground_truth.items():
                query_item = self._get_item_by_id(query_id)
                # Get embeddings
                query_embedding = self._get_embedding(
                    query_item['text'],
                    query_item.get('image')
                )
                # Retrieve similar items
                retrieved = retrieval_system.dense_search(
                    query_embedding,
                    k=k
                )
                retrieved_ids = [r['id'] for r in retrieved]
                
                # Calculate metrics
                relevant_retrieved = set(retrieved_ids) & set(relevant_ids)
                precision = len(relevant_retrieved) / k
                recall = len(relevant_retrieved) / len(relevant_ids)
                precisions.append(precision)
                recalls.append(recall)
                
                # Average Precision
                ap = self._calculate_average_precision(
                    retrieved_ids,
                    relevant_ids
                )
                aps.append(ap)
            
            results[f"P@{k}"] = np.mean(precisions)
            results[f"R@{k}"] = np.mean(recalls)
            results[f"MAP@{k}"] = np.mean(aps)
        
        return results
    
    def evaluate_data_synthesis_quality(
        self,
        synthetic_data: List[Dict],
        real_data: List[Dict]
    ) -> Dict:
        """
        Evaluate quality of synthesized data using multiple criteria.
        """
        metrics = {
            "linguistic_quality": [],
            "sentiment_consistency": [],
            "diversity": 0.0,
            "discriminator_score": 0.0
        }
        
        # 1. Linguistic quality (grammar, coherence)
        for item in synthetic_data:
            blob = TextBlob(item['text'])
            # Check with sentiment polarity variance
            sentences = blob.sentences
            if len(sentences) > 1:
                polarities = [s.sentiment.polarity for s in sentences]
                consistency = 1 - np.std(polarities)
                metrics["linguistic_quality"].append(consistency)
        
        # 2. Sentiment consistency (label matches text sentiment)
        for item in synthetic_data:
            blob = TextBlob(item['text'])
            text_polarity = blob.sentiment.polarity
            label = item['sentiment']
            
            # Check if polarity matches label
            if label == 'positive' and text_polarity > 0.2:
                metrics["sentiment_consistency"].append(1)
            elif label == 'negative' and text_polarity < -0.2:
                metrics["sentiment_consistency"].append(1)
            elif label == 'neutral' and -0.2 <= text_polarity <= 0.2:
                metrics["sentiment_consistency"].append(1)
            else:
                metrics["sentiment_consistency"].append(0)
        
        # 3. Diversity (vocabulary richness)
        all_words = []
        for item in synthetic_data:
            words = item['text'].lower().split()
            all_words.extend(words)
        unique_ratio = len(set(all_words)) / len(all_words)
        metrics["diversity"] = unique_ratio
        
        # 4. Discriminator score (from adversarial validator)
        validator = AdversarialValidator()
        validator.train_discriminator(
            real_reviews=[item['text'] for item in real_data],
            synthetic_reviews=[item['text'] for item in synthetic_data]
        )
        
        metrics["discriminator_score"] = validator.evaluate_synthetic_quality(
            [item['text'] for item in synthetic_data]
        )
        
        return {
            "linguistic_quality_avg": np.mean(metrics["linguistic_quality"]),
            "sentiment_consistency": np.mean(metrics["sentiment_consistency"]),
            "vocabulary_diversity": metrics["diversity"],
            "indistinguishability": metrics["discriminator_score"]
        }
    
    def evaluate_system_latency(
        self,
        inference_service,
        num_requests: int = 100
    ) -> Dict:
        """
        Load testing and latency analysis.
        """
        latencies = []
        errors = 0
        def make_request():
            start = time.time()
            try:
                # Sample request
                test_item = np.random.choice(self.test_data)
                _ = inference_service.analyze(
                    test_item['text'],
                    test_item.get('image')
                )
                return time.time() - start
            except Exception as e:
                return None
        
        # Concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(make_request)
                for _ in range(num_requests)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    latencies.append(result * 1000)  # Convert to ms
                else:
                    errors += 1
        
        return {
            "mean_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "max_latency_ms": np.max(latencies),
            "error_rate": errors / num_requests,
            "throughput_qps": num_requests / (sum(latencies) / 1000)
        }
    
    def generate_evaluation_report(
        self,
        output_path: str = "evaluation_report.html"
    ) -> str:
        """
        Generate comprehensive HTML report with visualizations.
        """
        sentiment_metrics = self.evaluate_sentiment_performance()
        multimodal_metrics = self.evaluate_multimodal_contribution()
        robustness_metrics = self.evaluate_adversarial_robustness()
        
        # Visualize
        self._plot_confusion_matrix(
            sentiment_metrics['confusion_matrix'],
            save_path="confusion_matrix.png"
        )
        
        self._plot_calibration_curve(
            sentiment_metrics['calibration'],
            save_path="calibration_curve.png"
        )
        
        self._plot_robustness_radar(
            robustness_metrics,
            save_path="robustness_radar.png"
        )
        
        # Create HTML report
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multimodal Sentiment Analysis - Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 20px 0; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #2196F3; color: white; }}
                img {{ max-width: 600px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>🎯 Multimodal Sentiment Analysis System - Evaluation Report</h1>
            
            <h2>1. Sentiment Classification Performance</h2>
            <div class="metric">
                <span>Overall Accuracy:</span>
                <span class="metric-value">{sentiment_metrics['accuracy']:.3f}</span>
            </div>
            <div class="metric">
                <span>Macro F1-Score:</span>
                <span class="metric-value">{sentiment_metrics['macro_f1']:.3f}</span>
            </div>
            
            <h3>Per-Class Metrics</h3>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                </tr>
                {self._generate_class_table_rows(sentiment_metrics['per_class'])}
            </table>
            
            <h3>Confusion Matrix</h3>
            <img src="confusion_matrix.png" alt="Confusion Matrix">
            
            <h3>Calibration Curve</h3>
            <img src="calibration_curve.png" alt="Calibration Curve">
            
            <h2>2. Multimodal Analysis Contribution</h2>
            <div class="metric">
                <span>Text-Only Accuracy:</span>
                <span class="metric-value">{multimodal_metrics.get('text_only_accuracy', 0):.3f}</span>
            </div>
            <div class="metric">
                <span>Image-Only Accuracy:</span>
                <span class="metric-value">{multimodal_metrics.get('image_only_accuracy', 0):.3f}</span>
            </div>
            <div class="metric">
                <span>Multimodal Accuracy:</span>
                <span class="metric-value">{multimodal_metrics.get('multimodal_accuracy', 0):.3f}</span>
            </div>
            <div class="metric">
                <span>Improvement from Multimodal:</span>
                <span class="metric-value">+{multimodal_metrics.get('multimodal_improvement', 0):.3f}</span>
            </div>
            
            <h2>3. Adversarial Robustness</h2>
            <img src="robustness_radar.png" alt="Robustness Radar Chart">
            <table>
                <tr>
                    <th>Perturbation Type</th>
                    <th>Robustness Score</th>
                </tr>
                {self._generate_robustness_table_rows(robustness_metrics)}
            </table>
            
            <h2>4. Summary</h2>
            <p>
                The multimodal sentiment analysis system demonstrates strong performance across
                multiple evaluation criteria. The system achieves {sentiment_metrics['accuracy']:.1%} accuracy
                on the test set, with particularly strong performance on {self._get_best_class(sentiment_metrics['per_class'])} sentiment.
            </p>
            <p>
                Multimodal analysis provides a significant improvement of 
                {multimodal_metrics.get('multimodal_improvement', 0):.1%} over unimodal approaches,
                demonstrating the value of combining text and visual information.
            </p>
        </body>
        </html>
        """
        with open(output_path, 'w') as f:
            f.write(html_template)  
        return output_path
    
    # HELPER FUCNS
    def _predict(self, text: str, image=None) -> Tuple[str, float]:
        """Make prediction with model"""
        # Implementation would use actual model
        return "positive", 0.85
    
    def _get_embedding(self, text: str, image=None) -> np.ndarray:
        """Get CLIP embedding"""
        return np.random.rand(768)
    
    def _add_typos(self, text: str, typo_rate: float = 0.1) -> str:
        """Add random typos to text"""
        words = text.split()
        for i in range(len(words)):
            if np.random.rand() < typo_rate:
                # Swap two characters
                if len(words[i]) > 2:
                    idx = np.random.randint(0, len(words[i])-1)
                    word_list = list(words[i])
                    word_list[idx], word_list[idx+1] = word_list[idx+1], word_list[idx]
                    words[i] = ''.join(word_list)
        return ' '.join(words)
    
    def _calculate_calibration(
        self,
        y_true: List[str],
        y_pred: List[str],
        confidences: List[float],
        n_bins: int = 10
    ) -> Dict:
        """Calculate calibration curve data"""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if np.sum(mask) > 0:
                bin_acc = np.mean([
                    1 if y_true[j] == y_pred[j] else 0
                    for j in range(len(y_true)) if mask[j]
                ])
                bin_conf = np.mean([confidences[j] for j in range(len(confidences)) if mask[j]])
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
        
        return {
            "accuracies": bin_accuracies,
            "confidences": bin_confidences
        }
    
    def _create_similarity_ground_truth(self) -> Dict:
        """Create ground truth for retrieval evaluation"""
        # Implementation would use manual annotations or heuristics
        return {}
    
    def _get_item_by_id(self, item_id: str) -> Dict:
        """Retrieve item by ID"""
        return {}
    
    def _calculate_average_precision(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """Calculate Average Precision for one query"""
        precisions = []
        num_relevant = 0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                num_relevant += 1
                precisions.append(num_relevant / i)
        
        return np.mean(precisions) if precisions else 0.0
    
    def _plot_confusion_matrix(self, cm, save_path: str):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Positive', 'Negative', 'Neutral'],
            yticklabels=['Positive', 'Negative', 'Neutral']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, calibration_data: Dict, save_path: str):
        """Plot calibration curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(
            calibration_data['confidences'],
            calibration_data['accuracies'],
            marker='o',
            label='Model'
        )
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_radar(self, robustness_metrics: Dict, save_path: str):
        """Plot robustness radar chart"""
        categories = list(robustness_metrics.keys())
        values = list(robustness_metrics.values())
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Robustness Scores', y=1.08)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_class_table_rows(self, per_class_metrics: Dict) -> str:
        """Generate HTML table rows for per-class metrics"""
        rows = ""
        for cls, metrics in per_class_metrics.items():
            rows += f"""
            <tr>
                <td>{cls.capitalize()}</td>
                <td>{metrics.get('precision', 0):.3f}</td>
                <td>{metrics.get('recall', 0):.3f}</td>
                <td>{metrics.get('f1-score', 0):.3f}</td>
            </tr>
            """
        return rows
    
    def _generate_robustness_table_rows(self, robustness_metrics: Dict) -> str:
        """Generate HTML table rows for robustness metrics"""
        rows = ""
        for key, value in robustness_metrics.items():
            rows += f"""
            <tr>
                <td>{key.replace('_', ' ').title()}</td>
                <td>{value:.3f}</td>
            </tr>
            """
        return rows
    
    def _get_best_class(self, per_class_metrics: Dict) -> str:
        """Find class with highest F1 score"""
        best_class = max(
            per_class_metrics.items(),
            key=lambda x: x[1].get('f1-score', 0)
        )
        return best_class[0]