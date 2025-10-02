import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from diffusers import StableDiffusionXLPipeline, ControlNetModel
from diffusers.utils import load_image
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI 
from PIL import Image
import random

class AdvancedSynthesisAgent:
    """
    Multi-modal synthesis agent that generates realistic, domain-specific
    review text and images with controllable sentiment signals.
    """
    
    def __init__(self, config: Dict):
        # Using mock implementation for demo
        print("Using mock AdvancedSynthesisAgent for demo")

    def synthesize_review_with_sentiment(
        self,
        target_sentiment: str,
        product_name: str,
        length: str = "medium"
    ) -> Tuple[str, Dict]:
        review_text = f"This is a mock {target_sentiment} review for {product_name}."
        metadata = { "synthetic": True, "target_sentiment": target_sentiment }
        return review_text, metadata

    def synthesize_product_image(
        self,
        product_name: str,
        sentiment_signal: str,
        condition_image: Image.Image = None
    ) -> Image.Image:
        # Return a blank image for the demo
        return Image.new('RGB', (100, 100))
    
#     def __init__(self, config: Dict):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # Text synthesis: Fine-tuned Llama 3.1
#         self.text_model = AutoModelForCausalLM.from_pretrained(
#             "meta-llama/Meta-Llama-3.1-8B-Instruct",
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             "meta-llama/Meta-Llama-3.1-8B-Instruct"
#         )
        
#         # Image synthesis: SDXL with ControlNet
#         controlnet = ControlNetModel.from_pretrained(
#             "diffusers/controlnet-canny-sdxl-1.0",
#             torch_dtype=torch.float16
#         )
        
#         self.image_pipeline = StableDiffusionXLPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-base-1.0",
#             controlnet=controlnet,
#             torch_dtype=torch.float16,
#             variant="fp16"
#         ).to(self.device)
        
#         # OpenAI for meta-reasoning and validation
#         self.openai = OpenAI(api_key=config['openai_api_key'])
        
#         # Domain knowledge extracted from seed data
#         self.domain_patterns = self._extract_domain_patterns(
#             config['seed_reviews']
#         )
        
#     def _extract_domain_patterns(self, seed_reviews: List[str]) -> Dict:
#         """
#         Analyze seed reviews to extract linguistic patterns, 
#         common phrases, and domain-specific vocabulary.
#         """
        
#         prompt = f"""Analyze these product reviews and extract:
# 1. Common phrases and expressions
# 2. Domain-specific terminology
# 3. Typical review structure patterns
# 4. Sentiment indicators (positive/negative phrases)

# Reviews:
# {chr(10).join(seed_reviews[:20])}

# Provide a JSON summary."""

#         response = self.openai.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=2048
#         )
        
#         patterns = eval(response.choices[0].message.content)  # Safe in controlled env
#         return patterns
    
#     def synthesize_review_with_sentiment(
#         self,
#         target_sentiment: str,  # 'positive', 'negative', 'neutral'
#         product_name: str,
#         length: str = "medium"  # 'short', 'medium', 'long'
#     ) -> Tuple[str, Dict]:
#         """
#         Generate a synthetic review with controlled sentiment.
#         Returns (review_text, metadata)
#         """
        
#         # Length mapping
#         length_map = {
#             "short": "1-2 sentences",
#             "medium": "3-5 sentences",
#             "long": "6-10 sentences"
#         }
        
#         # Sentiment-specific guidance
#         sentiment_guidance = {
#             "positive": "Focus on excellent quality, exceeded expectations, highly recommended",
#             "negative": "Mention specific flaws, disappointment, would not recommend",
#             "neutral": "Balanced perspective, some pros and cons, mixed feelings"
#         }
        
#         # Sample domain phrases for in-context learning
#         example_phrases = random.sample(
#             self.domain_patterns.get('common_phrases', []),
#             k=min(3, len(self.domain_patterns.get('common_phrases', [])))
#         )
        
#         prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id>
# You are an expert at writing authentic product reviews. Generate a {target_sentiment} review for {product_name}.

# Style guidelines:
# - Use natural, conversational language
# - Include specific details (not generic praise/complaints)
# - Incorporate domain phrases: {', '.join(example_phrases)}
# - Length: {length_map[length]}
# - Tone: {sentiment_guidance[target_sentiment]}

# <|eot_id|><|start_header_id|>user<|end_header_id>
# Write a {target_sentiment} review for {product_name}.<|eot_id|><|start_header_id|>assistant<|end_header_id>"""

#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
#         with torch.no_grad():
#             outputs = self.text_model.generate(
#                 **inputs,
#                 max_new_tokens=256,
#                 temperature=0.9,
#                 top_p=0.95,
#                 do_sample=True,
#                 repetition_penalty=1.2
#             )
        
#         review_text = self.tokenizer.decode(
#             outputs[0][inputs['input_ids'].shape[1]:],
#             skip_special_tokens=True
#         ).strip()
        
#         metadata = {
#             "synthetic": True,
#             "target_sentiment": target_sentiment,
#             "generation_method": "llama_3.1_finetuned",
#             "domain_patterns_used": example_phrases
#         }
        
#         return review_text, metadata
    
#     def synthesize_product_image(
#         self,
#         product_name: str,
#         sentiment_signal: str,
#         condition_image: Image.Image = None
#     ) -> Image.Image:
#         """
#         Generate product image with sentiment-appropriate visual cues.
        
#         For negative reviews: damaged, worn, poor lighting
#         For positive reviews: pristine, well-lit, professional
#         For neutral: average condition
#         """
        
#         sentiment_prompts = {
#             "positive": "professional product photography, pristine condition, excellent lighting, high quality, studio setup, clean background, sharp focus",
#             "negative": "damaged product, worn out, poor condition, scratches, dents, low quality, bad lighting, blurry",
#             "neutral": "average product photo, normal wear, standard lighting, casual setup"
#         }
        
#         base_prompt = f"{product_name}, {sentiment_prompts[sentiment_signal]}"
        
#         negative_prompt = "text, watermark, logo, distorted, low quality, blurry"
        
#         # Generate with sentiment-conditioned guidance
#         image = self.image_pipeline(
#             prompt=base_prompt,
#             negative_prompt=negative_prompt,
#             num_inference_steps=30,
#             guidance_scale=7.5,
#         ).images[0]
        
#         return image
    
#     def validate_synthetic_data(
#         self,
#         text: str,
#         image: Image.Image,
#         target_sentiment: str
#     ) -> Dict[str, float]:
#         """
#         Use adversarial validation to ensure synthetic data quality.
#         Returns quality scores.
#         """
        
#         # Text quality check via LLM
#         text_prompt = f"""Evaluate this product review on a scale of 0-1 for:
# 1. Authenticity (does it sound like a real person wrote it?)
# 2. Coherence (is it well-structured and logical?)
# 3. Sentiment alignment (does it match '{target_sentiment}'?)

# Review: {text}

# Return JSON: {{"authenticity": 0.X, "coherence": 0.X, "sentiment_alignment": 0.X}}"""

#         response = self.openai.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": text_prompt}],
#             max_tokens=256
#         )
        
#         text_scores = eval(response.choices[0].message.content)
        
#         # Image quality check (simplified - would use a discriminator network)
#         image_scores = {
#             "visual_quality": 0.85,  # Placeholder
#             "sentiment_visual_match": 0.80
#         }
        
#         overall_quality = np.mean([
#             text_scores['authenticity'],
#             text_scores['coherence'],
#             text_scores['sentiment_alignment'],
#             image_scores['visual_quality']
#         ])
        
#         return {
#             **text_scores,
#             **image_scores,
#             "overall_quality": overall_quality,
#             "passes_threshold": overall_quality > 0.75
#         }
    
#     def generate_balanced_dataset(self, target_size: int, product_names: List[str]) -> List[Dict]:
#         """
#         Generate a balanced synthetic dataset with controlled distribution.
#         """
        
#         sentiments = ['positive', 'negative', 'neutral']
#         per_sentiment = target_size // len(sentiments)
        
#         synthetic_data = []
        
#         for sentiment in sentiments:
#             for i in range(per_sentiment):
#                 product = random.choice(product_names)
                
#                 # Generate text
#                 text, text_meta = self.synthesize_review_with_sentiment(
#                     target_sentiment=sentiment,
#                     product_name=product,
#                     length=random.choice(['short', 'medium', 'long'])
#                 )
                
#                 # Generate corresponding image
#                 image = self.synthesize_product_image(
#                     product_name=product,
#                     sentiment_signal=sentiment
#                 )
                
#                 # Validate quality
#                 quality = self.validate_synthetic_data(text, image, sentiment)
                
#                 if quality['passes_threshold']:
#                     synthetic_data.append({
#                         "text": text,
#                         "image": image,
#                         "sentiment": sentiment,
#                         "product": product,
#                         "quality_scores": quality,
#                         "metadata": text_meta
#                     })
                    
#                     print(f"Generated {len(synthetic_data)}/{target_size} samples")
        
#         return synthetic_data
    
#     def augment_with_style_transfer(self, source_reviews: List[str], target_style: str = "formal") -> List[str]:
#         """
#         Apply style transfer to existing reviews to increase diversity (convert casual reviews to formal,...)
#         """
        
#         augmented = []
        
#         for review in source_reviews:
#             prompt = f"""Rewrite this review in a {target_style} style while preserving sentiment and key information:

# Original: {review}

# Rewritten ({target_style}):"""

#             response = self.openai.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=512
#             )
            
#             augmented.append(response.choices[0].message.content.strip())
        
#         return augmented


class AdversarialValidator:
    """
    Discriminator network to distinguish synthetic from real reviews.
    Ensures synthetic data is indistinguishable from real data.
    """
    
    def __init__(self):
        # Binary classifier: real vs synthetic
        self.discriminator = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.discriminator.to(self.device)
    
    def train_discriminator(
        self,
        real_reviews: List[str],
        synthetic_reviews: List[str],
        epochs: int = 3
    ):
        """
        Train discriminator to detect synthetic data.
        If discriminator accuracy > 0.7, synthetic data needs improvement.
        """
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Create dataset
        texts = real_reviews + synthetic_reviews
        labels = [0] * len(real_reviews) + [1] * len(synthetic_reviews)
        
        class ReviewDataset(Dataset):
            def __init__(self, texts, labels):
                self.encodings = tokenizer(
                    texts, 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
        
        dataset = ReviewDataset(texts, labels)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=2e-5)
        
        self.discriminator.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.discriminator(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    def evaluate_synthetic_quality(
        self,
        synthetic_reviews: List[str]
    ) -> float:
        """
        Lower discrimination accuracy = better synthetic data quality.
        Target: < 0.6 (almost indistinguishable from real)
        """
        
        self.discriminator.eval()
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        predictions = []
        with torch.no_grad():
            for review in synthetic_reviews:
                inputs = tokenizer(
                    review,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.discriminator(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(pred)
        
        # Proportion correctly identified as synthetic
        detection_rate = sum(predictions) / len(predictions)
        
        return 1 - detection_rate  # Quality score (higher is better)