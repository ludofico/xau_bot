
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoConfig
from typing import Optional, List
from xauusd_strategy.utils.logger import get_logger

logger = get_logger("TransformerEmbedder")

class TransformerEmbedder:
    """
    The 'Linguist' module: Extracts high-dimensional patterns from price action.
    Uses Amazon Chronos-Bolt (T5-based) to 'read' the chart.
    """
    
    def __init__(self, model_id: str = "amazon/chronos-bolt-tiny", device: str = "auto"):
        self.model_id = model_id
        if device == "auto" or device is None:
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        logger.info(f"Initializing TransformerEmbedder on {self.device}...")
        
        try:
            # We use the encoder part of Chronos to get hidden states (embeddings)
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
            self.model.eval()
            self.config = AutoConfig.from_pretrained(model_id)
            logger.info(f"Loaded {model_id} successfully.")
        except Exception as e:
            logger.error(f"Failed to load Transformer model: {e}")
            self.model = None

    def get_embeddings(self, series: np.ndarray, context_length: int = 64) -> Optional[np.ndarray]:
        """
        Generate embeddings for a univariate time series.
        
        Args:
            series: 1D array of prices (usually Close)
            context_length: Window size for the transformer
            
        Returns:
            Normalized embedding vector (numpy)
        """
        if self.model is None or len(series) < context_length:
            return None
            
        # Select last context_length points
        window = series[-context_length:]
        
        # Chronos expects (batch, sequence_length)
        # We normalize the window internally to handle massive price scales like Gold
        mean = window.mean()
        std = window.std() + 1e-8
        norm_window = (window - mean) / std
        
        # Convert to tensor
        # Chronos expects input as float tensor (batch, sequence)
        input_tensor = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                # Chronos-Bolt uses predict method for time series, not forward
                # But for embeddings, we can try to get encoder output
                if hasattr(self.model, 'encoder'):
                    # Quantize input for token embedding (Chronos uses input_ids internally)
                    # Fallback: use mean/std encoded representation
                    outputs = self.model.encoder(inputs_embeds=input_tensor.unsqueeze(-1))
                    embeddings = outputs.last_hidden_state
                else:
                    # Direct model call (may vary by architecture)
                    outputs = self.model(input_tensor)
                    if hasattr(outputs, 'last_hidden_state'):
                        embeddings = outputs.last_hidden_state
                    else:
                        embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Global Average Pooling over the time dimension to get a fixed-size vector
                pool_vec = torch.mean(embeddings, dim=1).squeeze().cpu().numpy()
                
                return pool_vec
        except Exception as e:
            logger.debug(f"Embedding extraction skipped: {e}")
            return None

    def add_transformer_features(self, df: pd.DataFrame, column: str = 'close', window: int = 64) -> pd.DataFrame:
        """
        Augment a DataFrame with transformer embeddings (heavy operation).
        NOTE: In live trading, only the LAST row is computed.
        """
        if self.model is None: return df
        
        # Get hidden dimension from model (e.g. 256 for tiny)
        hidden_dim = self.model.config.d_model if hasattr(self.model.config, 'd_model') else 256
        
        # In actual production, we don't calculate this for every row of a huge history
        # We only do it for the rows we need.
        logger.info(f"Generating embeddings for {len(df)} rows (Window={window})...")
        
        # Placeholder for final vector names - use pd.concat to avoid fragmentation
        feat_names = [f"trans_feat_{i}" for i in range(hidden_dim)]
        new_cols = pd.DataFrame(0.0, index=df.index, columns=feat_names)
        df = pd.concat([df, new_cols], axis=1)
        
        # Optimized: only do last N rows or some sample for backtesting
        # For real-time, just do df.iloc[-1]
        raw_values = df[column].values
        
        for i in range(window, len(df)):
            if i % 100 == 0: logger.debug(f"Processing row {i}/{len(df)}")
            vec = self.get_embeddings(raw_values[:i+1], context_length=window)
            if vec is not None:
                df.iloc[i, df.columns.get_indexer(feat_names)] = vec
                
        return df
