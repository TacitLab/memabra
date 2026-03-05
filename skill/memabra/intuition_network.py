"""
Intuition Network: Neural policy network for strategy selection.
PyTorch implementation with automatic differentiation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass  
class StrategyPrediction:
    """Output of intuition network."""
    strategy_id: str
    confidence: float
    memory_query_vector: List[float]
    all_scores: Dict[str, float]


class IntuitionNetwork(nn.Module):
    """
    神经网络直觉网络：将问题嵌入映射到策略选择 + 记忆查询向量。
    
    Architecture:
    - Input: query embedding (e.g., 384-dim)
    - Shared MLP: feature extraction with LayerNorm and Residual
    - Branch 1 (Strategy Head): strategy selection
    - Branch 2 (Memory Query Head): memory retrieval vector
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_strategies: int = 4,
        memory_query_dim: int = 384,
        dropout: float = 0.1,
        strategy_names: Optional[List[str]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies
        self.memory_query_dim = memory_query_dim
        self.lr = lr
        self.weight_decay = weight_decay
        
        # 策略名称映射
        self.strategy_names = strategy_names or [
            'direct_answer',
            'search_required',
            'tool_use',
            'clarification'
        ]
        assert len(self.strategy_names) == num_strategies
        
        # 共享特征提取器（带残差连接）
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # 隐藏层
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.hidden_dropout = nn.Dropout(dropout)
        
        # 策略选择头
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_strategies)
        )
        
        # 记忆查询向量生成头
        self.memory_query_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, memory_query_dim)
        )
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # 优化器（延迟初始化）
        self.optimizer = None
        
        # 训练统计
        self.training_stats = {
            'updates': 0,
            'total_reward': 0.0,
            'strategy_distribution': {s: 0 for s in self.strategy_names}
        }
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def setup_optimizer(self, lr: Optional[float] = None, weight_decay: Optional[float] = None):
        """设置优化器。"""
        lr = lr or self.lr
        weight_decay = weight_decay or self.weight_decay
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Args:
            x: 输入 embedding [batch_size, input_dim] 或 [input_dim]
            
        Returns:
            strategy_logits: 策略 logits
            memory_query: 记忆查询向量（已归一化）
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 输入投影
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.relu(h)
        h = self.input_dropout(h) if training else h
        
        # 残差块
        residual = h
        h = self.hidden(h)
        h = self.hidden_norm(h)
        h = F.relu(h)
        h = self.hidden_dropout(h) if training else h
        h = h + residual  # 残差连接
        
        # 策略预测
        strategy_logits = self.strategy_head(h)
        
        # 记忆查询向量（L2 归一化，便于余弦相似度计算）
        memory_query = self.memory_query_head(h)
        memory_query = F.normalize(memory_query, p=2, dim=-1)
        
        return strategy_logits, memory_query
    
    def predict(self, query_embedding: List[float]) -> StrategyPrediction:
        """
        预测最佳策略和记忆查询向量。
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(query_embedding, dtype=torch.float32)
            if not next(self.parameters()).is_cpu:
                x = x.to(next(self.parameters()).device)
            
            strategy_logits, memory_query = self.forward(x, training=False)
            
            # Temperature-scaled softmax
            temp = self.temperature.abs().clamp_min(0.1)
            probs = F.softmax(strategy_logits / temp, dim=-1)
            probs = probs.squeeze(0)
            
            # 选择最佳策略
            best_idx = torch.argmax(probs).item()
            confidence = probs[best_idx].item()
            strategy_id = self.strategy_names[best_idx]
            
            # 所有策略的分数
            all_scores = {
                self.strategy_names[i]: probs[i].item()
                for i in range(self.num_strategies)
            }
            
            return StrategyPrediction(
                strategy_id=strategy_id,
                confidence=confidence,
                memory_query_vector=memory_query.squeeze(0).cpu().tolist(),
                all_scores=all_scores
            )
    
    def update(
        self,
        query_embedding: List[float],
        strategy_id: str,
        reward: float,
        advantage: Optional[float] = None
    ) -> Dict[str, float]:
        """
        基于奖励更新网络（REINFORCE / Policy Gradient）。
        """
        if self.optimizer is None:
            self.setup_optimizer()
        
        self.train()
        self.optimizer.zero_grad()
        
        x = torch.tensor(query_embedding, dtype=torch.float32)
        if not next(self.parameters()).is_cpu:
            x = x.to(next(self.parameters()).device)
        
        strategy_logits, memory_query = self.forward(x, training=True)
        
        # 策略索引
        strategy_idx = self.strategy_names.index(strategy_id)
        
        # Log probability
        temp = self.temperature.abs().clamp_min(0.1)
        log_probs = F.log_softmax(strategy_logits / temp, dim=-1)
        log_prob = log_probs[0, strategy_idx]
        
        # REINFORCE loss
        target = reward if advantage is None else advantage
        policy_loss = -log_prob * target
        
        # 熵正则化（鼓励探索）
        probs = F.softmax(strategy_logits / temp, dim=-1)
        entropy = -(probs * log_probs).sum()
        entropy_bonus = 0.01 * entropy
        
        # 总损失
        total_loss = policy_loss - entropy_bonus
        
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新统计
        self.training_stats['updates'] += 1
        self.training_stats['total_reward'] += reward
        self.training_stats['strategy_distribution'][strategy_id] += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'log_prob': log_prob.item(),
            'temperature': self.temperature.item()
        }
    
    def update_memory_query(
        self,
        query_embedding: List[float],
        successful_memory_embedding: List[float],
        margin: float = 0.5
    ) -> float:
        """
        使用对比学习更新记忆查询生成器。
        让生成的查询向量更接近成功记忆的 embedding。
        """
        if self.optimizer is None:
            self.setup_optimizer()
        
        self.train()
        self.optimizer.zero_grad()
        
        x = torch.tensor(query_embedding, dtype=torch.float32)
        target = torch.tensor(successful_memory_embedding, dtype=torch.float32)
        
        if not next(self.parameters()).is_cpu:
            x = x.to(next(self.parameters()).device)
            target = target.to(next(self.parameters()).device)
        
        _, memory_query = self.forward(x, training=True)
        memory_query = memory_query.squeeze(0)
        target = target.squeeze(0)
        
        # 余弦相似度损失（最大化相似度 = 最小化 1 - similarity）
        similarity = F.cosine_similarity(memory_query.unsqueeze(0), target.unsqueeze(0))
        loss = 1 - similarity
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_strategy_weights(self) -> Dict[str, List[float]]:
        """
        获取策略的隐式权重（用于可解释性）。
        返回每个策略对应的原型向量。
        """
        self.eval()
        with torch.no_grad():
            # 提取策略头的最后一层权重作为策略原型
            final_layer = self.strategy_head[-1]
            weights = final_layer.weight.cpu().numpy()
            
            return {
                self.strategy_names[i]: weights[i].tolist()
                for i in range(self.num_strategies)
            }
    
    def save(self, path: str):
        """保存模型。"""
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_strategies': self.num_strategies,
                'memory_query_dim': self.memory_query_dim,
                'strategy_names': self.strategy_names,
                'lr': self.lr,
                'weight_decay': self.weight_decay
            },
            'stats': self.training_stats
        }, path)
    
    @classmethod
    def load(cls, path: str, map_location: str = 'cpu') -> 'IntuitionNetwork':
        """加载模型。"""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = checkpoint['config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        
        if checkpoint['optimizer']:
            model.setup_optimizer()
            model.optimizer.load_state_dict(checkpoint['optimizer'])
        
        model.training_stats = checkpoint['stats']
        return model
    
    def to_device(self, device: str):
        """移动模型到指定设备。"""
        return self.to(device)


class AdaptiveThreshold:
    """
    自适应置信度阈值：根据历史表现动态调整。
    """
    
    def __init__(
        self,
        initial: float = 0.7,
        min_threshold: float = 0.5,
        max_threshold: float = 0.9,
        window_size: int = 100
    ):
        self.threshold = initial
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.window_size = window_size
        self.history: List[Tuple[float, bool]] = []
    
    def should_explore(self, confidence: float) -> bool:
        """判断是否应该进入探索模式。"""
        return confidence < self.threshold
    
    def update(self, confidence: float, success: bool):
        """根据结果更新阈值。"""
        self.history.append((confidence, success))
        
        # 保持窗口大小
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        # 每10条记录调整一次
        if len(self.history) % 10 == 0:
            self._adjust_threshold()
    
    def _adjust_threshold(self):
        """调整阈值以优化准确率/召回率平衡。"""
        if len(self.history) < 10:
            return
        
        # 分析高置信度样本的表现
        high_conf_samples = [(c, s) for c, s in self.history if c >= self.threshold]
        if len(high_conf_samples) < 5:
            return
        
        high_conf_accuracy = sum(s for _, s in high_conf_samples) / len(high_conf_samples)
        
        # 如果高置信度准确率低于80%，提高阈值
        if high_conf_accuracy < 0.8:
            self.threshold = min(self.threshold + 0.05, self.max_threshold)
        # 如果高置信度准确率高于95%，可以降低阈值来增加覆盖
        elif high_conf_accuracy > 0.95:
            self.threshold = max(self.threshold - 0.02, self.min_threshold)
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息。"""
        if not self.history:
            return {'threshold': self.threshold, 'samples': 0}
        
        high_conf = [s for c, s in self.history if c >= self.threshold]
        low_conf = [s for c, s in self.history if c < self.threshold]
        
        return {
            'threshold': self.threshold,
            'total_samples': len(self.history),
            'high_conf_accuracy': sum(high_conf) / len(high_conf) if high_conf else 0,
            'low_conf_accuracy': sum(low_conf) / len(low_conf) if low_conf else 0,
        }


class ExplorationController:
    """
    探索模式控制器：低置信度时尝试多策略。
    """
    
    def __init__(self, intuition_net: IntuitionNetwork, threshold: float = 0.7):
        self.intuition_net = intuition_net
        self.threshold = threshold
        self.adaptive_threshold = AdaptiveThreshold(initial=threshold)
    
    def decide_path(
        self,
        query_embedding: List[float],
        force_explore: bool = False
    ) -> Tuple[str, StrategyPrediction, List[StrategyPrediction]]:
        """
        决定是走快速路径还是探索模式。
        
        Returns:
            path_type: 'fast' 或 'exploration'
            primary_prediction: 主要预测结果
            all_predictions: 所有候选策略（探索模式下填充）
        """
        primary = self.intuition_net.predict(query_embedding)
        
        if not force_explore and not self.adaptive_threshold.should_explore(primary.confidence):
            return 'fast', primary, [primary]
        
        # 探索模式：生成所有策略的预测
        self.intuition_net.eval()
        with torch.no_grad():
            x = torch.tensor(query_embedding, dtype=torch.float32)
            if not next(self.intuition_net.parameters()).is_cpu:
                x = x.to(next(self.intuition_net.parameters()).device)
            
            _, memory_query = self.intuition_net.forward(x, training=False)
            memory_query_vec = memory_query.squeeze(0).cpu().tolist()
        
        all_predictions = []
        for strategy_id in self.intuition_net.strategy_names:
            pred = StrategyPrediction(
                strategy_id=strategy_id,
                confidence=primary.all_scores.get(strategy_id, 0.0),
                memory_query_vector=memory_query_vec,
                all_scores=primary.all_scores
            )
            all_predictions.append(pred)
        
        # 按置信度排序
        all_predictions.sort(key=lambda p: p.confidence, reverse=True)
        
        return 'exploration', primary, all_predictions
    
    def report_outcome(self, confidence: float, success: bool):
        """报告结果以更新自适应阈值。"""
        self.adaptive_threshold.update(confidence, success)
        self.threshold = self.adaptive_threshold.threshold
