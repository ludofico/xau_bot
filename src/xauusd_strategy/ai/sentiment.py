
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional
from xauusd_strategy.utils.logger import get_logger
import re

logger = get_logger("SentimentAnalyst")

class SentimentAnalyst:
    """
    The 'Sentinel' module: Uses a local Micro LLM to analyze market sentiment.
    Default: TinyLlama-1.1B (Lightweight, fits in 2GB VRAM or CPU).
    """
    
    def __init__(self, model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = "auto"):
        self.model_id = model_id
        if device == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        logger.info(f"Initializing SentimentAnalyst on {self.device}...")
        
        try:
            # We use a 4-bit/8-bit quantized version if possible, but for simplicity here we load base
            # In a real setup, we'd use llama-cpp for speed.
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            logger.info("Local Micro LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Sentiment LLM: {e}")
            self.model = None

    def analyze_headline(self, headline: str) -> float:
        """
        Analyze a single financial headline.
        Returns a Risk Multiplier: 
        1.0 (Safe), 0.5 (Caution), 0.0 (High Risk/Halt)
        """
        if self.model is None: return 1.0
        
        prompt = f"<|system|>\nYou are a financial risk analyst. Categorize the impact of this news on Gold (XAUUSD) trading as: HIGH, MEDIUM, or LOW.\n<|user|>\nHeadline: {headline}\n<|assistant|>\nImpact:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            # Extract category
            result = response.split("Impact:")[-1].strip().upper()
            logger.debug(f"Sentiment Result: {result} for '{headline}'")
            
            if "HIGH" in result:
                return 0.0 # HALT
            elif "MEDIUM" in result:
                return 0.5 # REDUCE RISK
            else:
                return 1.0 # NORMAL
        except Exception as e:
            logger.error(f"Sentiment inference error: {e}")
            return 1.0

    def get_news_risk_multiplier(self, headlines: List[str]) -> float:
        """Aggregate risks from multiple sources."""
        if not headlines: return 1.0
        
        multipliers = [self.analyze_headline(h) for h in headlines]
        # Return the most conservative (minimum) multiplier found
        return min(multipliers)
