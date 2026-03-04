"""
Memabra: Bio-inspired memory and intuition system for AI agents.

核心组件：
- IntuitionNetwork: 神经网络直觉网络，用于策略选择和记忆查询向量生成
- ImplicitEvaluator: 隐式反馈评估器，从用户行为中推断满意度
- HierarchicalMemory: 分层记忆系统（情景/语义/程序）
- DelayedRewardAssigner: 延迟奖励分配器
"""

__version__ = "0.1.0"

from .intuition_network import (
    IntuitionNetwork,
    StrategyPrediction,
    AdaptiveThreshold,
    ExplorationController
)
from .feedback_evaluator import (
    ImplicitEvaluator,
    FeedbackSignal,
    SignalType,
    Interaction,
    DelayedRewardAssigner,
    FeedbackCalibrator
)
from .memory import (
    HierarchicalMemory,
    Memory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    EpisodicStore,
    SemanticStore,
    ProceduralStore
)
from .agent import MemabraAgent

__all__ = [
    # 直觉网络
    "IntuitionNetwork",
    "StrategyPrediction",
    "AdaptiveThreshold",
    "ExplorationController",
    
    # 反馈系统
    "ImplicitEvaluator",
    "FeedbackSignal",
    "SignalType",
    "Interaction",
    "DelayedRewardAssigner",
    "FeedbackCalibrator",
    
    # 记忆系统
    "HierarchicalMemory",
    "Memory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "EpisodicStore",
    "SemanticStore",
    "ProceduralStore",
    
    # 完整代理
    "MemabraAgent",
]
