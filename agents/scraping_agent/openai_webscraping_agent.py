import asyncio
import os
from datetime import datetime
import hashlib
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
import boto3
from dotenv import load_dotenv
from openai import OpenAI 
from playwright.async_api import async_playwright
import re


@dataclass
class ReviewData:
    """Structured review data"""
    text: str
    image_urls: List[str]
    rating: Optional[float]
    product_name: str
    source_url: str
    timestamp: datetime
    metadata: Dict


class IntelligentScrapingAgent:
    """
    Autonomous agent that learns which sources are valuable
    and adapts its scraping strategy using LLM-guided decisions.
    """

    def __init__(self, openai_api_key: str, aws_config: Dict):
        self.openai = OpenAI(api_key=openai_api_key)
        
        # Connect to local MinIO S3
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_config.get('aws_access_key_id'),
            aws_secret_access_key=aws_config.get('aws_secret_access_key'),
            region_name=aws_config.get('region_name'),
            endpoint_url=aws_config.get('endpoint_url')
        )
        self.bucket_name = "multimodal-reviews-raw"
        self.strategy_memory = self._load_strategy_memory()

    def _load_strategy_memory(self) -> Dict:
        """Load previous scraping success rates and strategies"""
        try:
            obj = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key='strategy_memory.json'
            )
            return json.loads(obj['Body'].read())
        except:
            return {
                "successful_sites": {},
                "failed_sites": {},
                "optimal_selectors": {}
            }

    async def decide_next_action(self, current_state: Dict) -> Dict:
        """
        Use OpenAI GPT to decide the next scraping action based on state.
        """
        prompt = f"""You are an intelligent web scraping strategist. Based on the following state, decide the next best action.

Current State:
- Total reviews collected: {current_state['total_reviews']}
- Quality score (0-1): {current_state['quality_score']}
- Source diversity: {current_state['unique_sources']} unique domains
- Failed attempts: {len(self.strategy_memory['failed_sites'])}

Strategy Memory:
{json.dumps(self.strategy_memory.get('successful_sites', {}), indent=2)[:500]}

Goal: Collect 10,000 high-quality reviews with diverse perspectives.

Provide your decision in a JSON object format.
"""
        
        response = self.openai.chat.completions.create(
            model="gpt-4o",  # Using a powerful model for reasoning
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a web scraping strategist that responds with only a valid JSON object based on the user's prompt."},
                {"role": "user", "content": prompt}
            ]
        )
        
        decision = json.loads(response.choices[0].message.content)
        return decision

    async def scrape_with_playwright(
        self,
        url: str,
        product_keywords: List[str]
    ) -> List[ReviewData]:
        """
        Use Playwright for JavaScript-heavy sites.
        Dynamically adapts to site structure.
        """
        reviews = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until='networkidle')
                await page.wait_for_selector('.review, [class*="review"]', timeout=5000)
                
                selectors = await self._discover_review_selectors(page)
                for selector in selectors:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        review_text = await element.text_content()
                        img_elements = await element.query_selector_all('img')
                        image_urls = [await img.get_attribute('src') for img in img_elements]
                        rating = await self._extract_rating(element)
                        
                        reviews.append(ReviewData(
                            text=review_text.strip() if review_text else "",
                            image_urls=image_urls,
                            rating=rating,
                            product_name=self._extract_product_name(url),
                            source_url=url,
                            timestamp=datetime.now(),
                            metadata={"selector": selector}
                        ))
            except Exception as e:
                print(f"Scraping error on {url}: {e}")
                self._update_failed_sites(url, str(e))
            finally:
                await browser.close()
        return reviews

    async def _discover_review_selectors(self, page) -> List[str]:
        """
        Intelligently discover review selectors using OpenAI GPT.
        """
        html_sample = await page.content()
        html_sample = html_sample[:5000]

        prompt = f"""Analyze this HTML and suggest CSS selectors for product review container elements. Return a JSON array of strings, with the most likely selectors first.

HTML Sample:
{html_sample}
"""
        
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": 'You are an expert front-end developer. Respond with a JSON object {"selectors": ["selector1", "selector2"]}.'},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("selectors", [])

    async def _extract_rating(self, element) -> Optional[float]:
        """Extract star rating or numerical score"""
        text = await element.inner_text()
        patterns = [r'(\d+\.?\d*)\s*(?:out of|\/)\s*5', r'(\d+\.?\d*)\s*stars?']
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_product_name(self, url: str) -> str:
        return url.split('/')[-1].replace('-', ' ')

    async def store_reviews(self, reviews: List[ReviewData]):
        """Store scraped reviews in S3 with deduplication"""
        for review in reviews:
            if not review.text: continue
            content_hash = hashlib.sha256(review.text.encode()).hexdigest()[:16]
            key = f"raw/{datetime.now().strftime('%Y-%m-%d')}/{content_hash}.json"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps({
                    "text": review.text,
                    "image_urls": review.image_urls,
                    "rating": review.rating,
                    "product_name": review.product_name,
                    "source_url": review.source_url,
                    "timestamp": review.timestamp.isoformat(),
                    "metadata": review.metadata
                }, indent=2),
                ContentType='application/json'
            )

    def _update_failed_sites(self, url: str, error: str):
        """Update strategy memory with failed attempts"""
        domain = url.split('/')[2]
        self.strategy_memory['failed_sites'][domain] = {
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self._persist_strategy_memory()

    def _persist_strategy_memory(self):
        """Save strategy memory to S3"""
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key='strategy_memory.json',
            Body=json.dumps(self.strategy_memory, indent=2)
        )


async def run_ingestion_pipeline(config: Dict):
    """ Main pipeline that orchestrates intelligent scraping """
    agent = IntelligentScrapingAgent(
        openai_api_key=config['openai_api_key'], 
        aws_config=config['aws']
    )
    current_state = {"total_reviews": 0, "quality_score": 0.7, "unique_sources": 0}

    # Example loop for 1 iteration
    print("Agent deciding next action...")
    decision = await agent.decide_next_action(current_state)
    print(f"Decision: {decision}")
    
    # Demo
    target_url = "https://www.goodreads.com/review/list/1?sort=popularity" # Goodreads user reviews
    print(f"Overriding target for demonstration. Scraping: {target_url}")

    if decision.get('action') != 'stop':
        reviews = await agent.scrape_with_playwright(
            url=target_url,
            product_keywords=config['product_keywords']
        )
        if reviews:
            await agent.store_reviews(reviews)
            current_state['total_reviews'] += len(reviews)
            print(f"Successfully collected and stored {len(reviews)} reviews from {target_url}")
        else:
            print(f"No reviews found on {target_url}")

if __name__ == "__main__":
    load_dotenv()
    
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"), 
        "aws": {
            "region_name": "us-east-1",
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "endpoint_url": "http://minio:9000"
        },
        "product_keywords": ["book", "novel"]
    }
    
    asyncio.run(run_ingestion_pipeline(config))