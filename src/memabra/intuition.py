"""
Intuition Network: Fast pattern recognition for strategy selection.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


class SimpleIntuitionNetwork:
    """
    Lightweight intuition network based on cosine similarity.
    
    Simulates human "System 1" thinking - fast, automatic, pattern-based decision making.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        
        # Strategy prototypes - each strategy is represented by a vector
        self.strategies = {
            'direct_answer': np.random.randn(embedding_dim),
            'search_required': np.random.randn(embedding_dim),
            'tool_use': np.random.randn(embedding_dim),
            'clarification': np.random.randn(embedding_dim),
        }
        
        # Track rewards for each strategy
        self.strategy_rewards = {k: [] for k in self.strategies}
        
        # Normalize initial vectors
        for sid in self.strategies:
            self.strategies[sid] = self._normalize(self.strategies[sid])
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def predict(self, problem_embedding: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict the best strategy for a given problem.
        
        Args:
            problem_embedding: Vector representation of the problem
            
        Returns:
            strategy_id: Selected strategy
            confidence: Confidence score (0-1)
            similarities: Similarity scores for all strategies
        """
        problem_emb = np.array(problem_embedding).reshape(1, -1)
        
        similarities = {}
        for sid, strat_emb in self.strategies.items():
            sim = cosine_similarity(problem_emb, strat_emb.reshape(1, -1))[0][0]
            # Convert to 0-1 range using sigmoid-like transformation
            similarities[sid] = float((sim + 1) / 2)
        
        best_strategy = max(similarities, key=similarities.get)
        confidence = similarities[best_strategy]
        
        return best_strategy, confidence, similarities
    
    def update(self, problem_embedding: np.ndarray, strategy_id: str, 
               reward: float, lr: float = 0.01) -> None:
        """
        Update strategy vector based on reward (online learning).
        
        Args:
            problem_embedding: The problem that led to this strategy
            strategy_id: Strategy that was used
            reward: Feedback signal (-1 to +1)
            lr: Learning rate
        """
        if strategy_id not in self.strategies:
            return
        
        problem_emb = np.array(problem_embedding)
        current = self.strategies[strategy_id]
        
        # Move towards problem vector (positive reward) or away (negative reward)
        new_vector = current + lr * reward * self._normalize(problem_emb)
        self.strategies[strategy_id] = self._normalize(new_vector)
        
        # Track reward history
        self.strategy_rewards[strategy_id].append(reward)
    
    def get_strategy_stats(self) -> Dict[str, Dict]:
        """Get statistics about each strategy's performance."""
        stats = {}
        for sid, rewards in self.strategy_rewards.items():
            if rewards:
                stats[sid] = {
                    'count': len(rewards),
                    'avg_reward': np.mean(rewards),
                    'recent_avg': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                }
            else:
                stats[sid] = {'count': 0, 'avg_reward': 0, 'recent_avg': 0}
        return stats


class NeuralIntuitionNet:
    """
    Neural network-based intuition network (placeholder for Phase 2).
    """
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, num_strategies: int = 4):
        self.input_dim = input_dim
        self.num_strategies = num_strategies
        # Actual implementation would use PyTorch
        raise NotImplementedError("NeuralIntuitionNet will be implemented in Phase 2")
