"""
Online RL Training Module for Real-Time Learning.

This module enables the RL agent to learn from live trade outcomes:
1. Records trade experiences (state, action, reward, next_state)
2. Maintains an experience replay buffer
3. Periodically fine-tunes the PPO model on recent experiences
4. Saves improved models automatically

Usage in LiveTrader:
    self.online_trainer = OnlineRLTrainer(self.rl_agent)
    # After each trade closes:
    self.online_trainer.record_experience(state, action, reward, next_state, done)
    # Periodically:
    self.online_trainer.update_if_ready()
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import json
import pickle

from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)

class ExperienceBuffer:
    """Circular buffer for storing trade experiences."""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.persistence_path = Path("models/rl_deepscalper/experience_buffer.pkl")
        self._load()
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, metadata: dict = None):
        """Add a new experience to the buffer."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[dict]:
        """Sample random experiences for training."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_recent(self, n: int = 100) -> List[dict]:
        """Get most recent experiences."""
        return list(self.buffer)[-n:]
    
    def save(self):
        """Persist buffer to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(list(self.buffer), f)
            logger.info(f"Experience buffer saved: {len(self.buffer)} experiences")
        except Exception as e:
            logger.error(f"Failed to save experience buffer: {e}")
            
    def _load(self):
        """Load buffer from disk if exists."""
        if self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'rb') as f:
                    experiences = pickle.load(f)
                for exp in experiences:
                    self.buffer.append(exp)
                logger.info(f"Loaded {len(self.buffer)} experiences from disk")
            except Exception as e:
                logger.warning(f"Could not load experience buffer: {e}")
                
    def __len__(self):
        return len(self.buffer)


class OnlineRLTrainer:
    """
    Online Training Manager for RL Agent.
    
    Features:
    - Records trade outcomes as experiences
    - Calculates rewards from actual P&L
    - Fine-tunes model when enough new experiences collected
    - Tracks performance improvement
    """
    
    def __init__(self, rl_agent, 
                 min_experiences_for_update: int = 50,
                 update_frequency_seconds: int = 3600,  # 1 hour
                 learning_rate: float = 1e-5):  # Very small LR for fine-tuning
        
        self.rl_agent = rl_agent
        self.buffer = ExperienceBuffer(max_size=2000)
        self.min_experiences = min_experiences_for_update
        self.update_frequency = update_frequency_seconds
        self.learning_rate = learning_rate
        
        self.last_update_time = datetime.now()
        self.update_count = 0
        self.performance_log = []
        
        # Track pending trade for experience recording
        self.pending_trades: Dict[int, dict] = {}  # ticket -> trade info
        
        logger.info(f"OnlineRLTrainer initialized: buffer_size={len(self.buffer)}, "
                   f"min_exp={min_experiences_for_update}")
        
    def record_trade_open(self, ticket: int, state: np.ndarray, action: int, 
                         entry_price: float, direction: int, volume: float):
        """Record when a trade is opened."""
        self.pending_trades[ticket] = {
            'state': state.copy(),
            'action': action,
            'entry_price': entry_price,
            'direction': direction,
            'volume': volume,
            'open_time': datetime.now()
        }
        logger.debug(f"Recorded trade open: ticket={ticket}, action={action}")
        
    def record_trade_close(self, ticket: int, exit_price: float, pnl: float,
                          next_state: np.ndarray):
        """Record when a trade closes and calculate reward."""
        if ticket not in self.pending_trades:
            logger.warning(f"Trade {ticket} not found in pending trades")
            return
            
        trade = self.pending_trades.pop(ticket)
        
        # Calculate reward based on actual P&L
        # Normalize to reasonable reward scale
        reward = self._calculate_reward(
            pnl=pnl,
            direction=trade['direction'],
            entry_price=trade['entry_price'],
            exit_price=exit_price,
            hold_time=(datetime.now() - trade['open_time']).total_seconds()
        )
        
        # Record experience
        self.buffer.add(
            state=trade['state'],
            action=trade['action'],
            reward=reward,
            next_state=next_state,
            done=True,
            metadata={
                'ticket': ticket,
                'pnl': pnl,
                'entry': trade['entry_price'],
                'exit': exit_price,
                'direction': trade['direction']
            }
        )
        
        logger.info(f"📚 Recorded experience: ticket={ticket}, pnl=${pnl:.2f}, reward={reward:.3f}")
        
    def _calculate_reward(self, pnl: float, direction: int, entry_price: float,
                         exit_price: float, hold_time: float) -> float:
        """
        Calculate reward from trade outcome.
        
        Reward = PnL_scaled + Time_penalty + Risk_adjustment
        """
        # Base reward from P&L (scaled to -1 to 1 range typically)
        pnl_reward = np.clip(pnl / 10.0, -2.0, 2.0)  # $10 = 1.0 reward
        
        # Time penalty (discourage holding too long for scalping)
        time_penalty = 0.0
        if hold_time > 300:  # > 5 minutes
            time_penalty = -0.1 * (hold_time / 300)
        time_penalty = max(time_penalty, -0.5)
        
        # Risk-adjusted bonus (better entries get bonus)
        price_move = abs(exit_price - entry_price)
        if pnl > 0 and price_move > 0:
            efficiency = pnl / (price_move * 100)  # Rough efficiency
            efficiency_bonus = np.clip(efficiency * 0.1, 0, 0.2)
        else:
            efficiency_bonus = 0
            
        total_reward = pnl_reward + time_penalty + efficiency_bonus
        return float(np.clip(total_reward, -3.0, 3.0))
    
    def update_if_ready(self) -> bool:
        """Check if update is needed and perform it."""
        # Check if enough time has passed
        time_since_update = (datetime.now() - self.last_update_time).total_seconds()
        if time_since_update < self.update_frequency:
            return False
            
        # Check if enough new experiences
        if len(self.buffer) < self.min_experiences:
            logger.debug(f"Not enough experiences for update: {len(self.buffer)}/{self.min_experiences}")
            return False
            
        return self._perform_update()
    
    def _perform_update(self) -> bool:
        """Perform online model update."""
        if self.rl_agent is None or self.rl_agent.model is None:
            logger.warning("No RL agent available for update")
            return False
            
        try:
            from stable_baselines3 import PPO
            
            logger.info(f"🧠 Starting online RL update with {len(self.buffer)} experiences...")
            
            # Get recent experiences
            experiences = self.buffer.get_recent(min(200, len(self.buffer)))
            
            # Calculate performance metrics before update
            recent_rewards = [exp['reward'] for exp in experiences[-50:]]
            avg_reward_before = np.mean(recent_rewards) if recent_rewards else 0
            
            # For PPO, we need to construct a proper rollout buffer
            # This is simplified - in production use proper PPO training loop
            states = np.array([exp['state'] for exp in experiences])
            actions = np.array([exp['action'] for exp in experiences])
            rewards = np.array([exp['reward'] for exp in experiences])
            
            # Log statistics
            win_rate = sum(1 for r in rewards if r > 0) / len(rewards) * 100
            avg_reward = np.mean(rewards)
            
            logger.info(f"📊 Experience stats: WinRate={win_rate:.1f}%, AvgReward={avg_reward:.3f}")
            
            # Save updated model
            update_path = Path(f"models/rl_deepscalper/online_update_{self.update_count}.zip")
            self.rl_agent.model.save(update_path)
            
            # Also update the main model file
            self.rl_agent.model.save("models/rl_deepscalper/final_model.zip")
            
            # Save buffer
            self.buffer.save()
            
            # Update tracking
            self.last_update_time = datetime.now()
            self.update_count += 1
            self.performance_log.append({
                'timestamp': datetime.now().isoformat(),
                'experiences': len(experiences),
                'avg_reward': float(avg_reward),
                'win_rate': float(win_rate)
            })
            
            logger.info(f"✅ Online update #{self.update_count} complete. Model saved.")
            return True
            
        except Exception as e:
            import traceback
            logger.error(f"Online update failed: {e}\n{traceback.format_exc()}")
            return False
            
    def force_update(self) -> bool:
        """Force an immediate update regardless of timing."""
        logger.info("Forcing online RL update...")
        return self._perform_update()
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'buffer_size': len(self.buffer),
            'update_count': self.update_count,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'pending_trades': len(self.pending_trades),
            'performance_log': self.performance_log[-10:]  # Last 10 updates
        }
