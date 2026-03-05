"""
Memabra: Bio-inspired memory and intuition system for AI agents.

OpenClaw Skill runtime modules:
- IntuitionNetwork: Neural policy network for strategy selection + memory query generation
- ImplicitEvaluator: Implicit feedback evaluator from user behavior
- HierarchicalMemory: Hierarchical memory system (episodic / semantic / procedural)
- MemabraAgent: Core agent integrating all modules
"""

__version__ = "0.2.0"

from .intuition_network import (
    IntuitionNetwork,
    StrategyPrediction,
    AdaptiveThreshold,
    ExplorationController,
)
from .feedback_evaluator import (
    ImplicitEvaluator,
    FeedbackSignal,
    SignalType,
    Interaction,
    DelayedRewardAssigner,
    FeedbackCalibrator,
)
from .memory import (
    HierarchicalMemory,
    Memory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    ActionMemory,
    ActionStep,
    EpisodicStore,
    SemanticStore,
    ProceduralStore,
    ActionStore,
)
from .agent import MemabraAgent

__all__ = [
    "IntuitionNetwork",
    "StrategyPrediction",
    "AdaptiveThreshold",
    "ExplorationController",
    "ImplicitEvaluator",
    "FeedbackSignal",
    "SignalType",
    "Interaction",
    "DelayedRewardAssigner",
    "FeedbackCalibrator",
    "HierarchicalMemory",
    "Memory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "ActionMemory",
    "ActionStep",
    "EpisodicStore",
    "SemanticStore",
    "ProceduralStore",
    "ActionStore",
    "MemabraAgent",
]
