
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from xauusd_strategy.utils.logger import get_logger

logger = get_logger("TickModel")

class TickLSTM(nn.Module):
    """Simple LSTM for HFT Tick Prediction."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2):
        super(TickLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1) # Predict Next Tick Delta
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class TickPredictor:
    """
    Wrapper for the LSTM Tick Model.
    Predicts if the next 5 ticks will be Up or Down.
    """
    def __init__(self, model_path: str = "models/tick_lstm.pth"):
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TickLSTM().to(self.device).to(torch.float32)
        self.model_path = model_path
        self._load()
        
    def _load(self):
        try:
            import os
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                logger.info("Tick LSTM loaded.")
            else:
                logger.warning("Tick LSTM weights not found. Using untrained model for architecture demo.")
        except Exception as e:
            logger.error(f"Tick model load error: {e}")

    def predict_delta(self, tick_history: List[Tuple[float, float]]) -> float:
        """
        Input: list of (price, volume) tuples for last 20 ticks.
        Output: predicted price change.
        """
        if len(tick_history) < 20: return 0.0
        
        # Normalize
        data = np.array(tick_history[-20:])
        prices = data[:, 0]
        vols = data[:, 1]
        
        # Calculate deltas
        price_deltas = np.diff(prices, prepend=prices[0])
        norm_vols = vols / (vols.mean() + 1e-8)
        
        # (Seq, Feat)
        x = np.stack([price_deltas, norm_vols], axis=1).astype(np.float32)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(x_tensor).item()
            
        return prediction
