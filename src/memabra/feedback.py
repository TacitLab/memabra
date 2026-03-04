"""
Implicit Feedback System: Infer user satisfaction from natural behavior.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class SignalType(Enum):
    """Types of feedback signals."""
    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    IMPLICIT_REPEAT = "implicit_repeat"
    IMPLICIT_REPHRASE = "implicit_rephrase"
    IMPLICIT_FOLLOWUP = "implicit_followup"
    IMPLICIT_TOPIC_SHIFT = "topic_shift"
    IMPLICIT_SILENCE = "silence"


@dataclass
class FeedbackSignal:
    """A single feedback signal with metadata."""
    signal_type: SignalType
    reward: float
    confidence: float
    explanation: str


class ImplicitEvaluator:
    """
    Evaluates user satisfaction from implicit signals in conversation.
    
    No explicit thumbs up/down required - we observe natural behavior.
    """
    
    # Chinese keywords for quick detection
    POSITIVE_KEYWORDS = {'谢谢', '可以', 'ok', '好的', '没错', '对', '完美', '搞定了', '很好'}
    NEGATIVE_KEYWORDS = {'不对', '错了', '不是', '不行', '没用', '不理解', '算了', '放弃'}
    CLARIFICATION_PATTERNS = {'为什么', '怎么', '那', '还有', '另外', '等等'}
    
    def __init__(self, embedding_fn=None, similarity_threshold: float = 0.75):
        self.embed = embedding_fn
        self.similarity_threshold = similarity_threshold
    
    def evaluate(self, 
                 last_assistant_msg: str, 
                 next_user_msg: str,
                 conversation_history: Optional[List[Dict]] = None) -> FeedbackSignal:
        """
        Evaluate the quality of last_assistant_msg based on user's next response.
        
        Args:
            last_assistant_msg: The assistant's last message
            next_user_msg: User's subsequent message
            conversation_history: Optional full conversation context
            
        Returns:
            FeedbackSignal with reward and confidence
        """
        # 1. Check explicit keywords first
        keyword_signal = self._check_keywords(next_user_msg)
        if keyword_signal:
            return keyword_signal
        
        # 2. Semantic similarity analysis
        if self.embed:
            sim_score = self._semantic_similarity(last_assistant_msg, next_user_msg)
            if sim_score > self.similarity_threshold:
                return FeedbackSignal(
                    signal_type=SignalType.IMPLICIT_REPEAT,
                    reward=-0.5,
                    confidence=0.8,
                    explanation=f"User repeated similar question (similarity: {sim_score:.2f})"
                )
        
        # 3. Analyze intent
        intent = self._analyze_intent(next_user_msg)
        
        if intent == 'clarification':
            return FeedbackSignal(
                signal_type=SignalType.IMPLICIT_FOLLOWUP,
                reward=-0.1,
                confidence=0.6,
                explanation="User asked for clarification/details"
            )
        
        elif intent == 'topic_shift':
            satisfaction = self._infer_satisfaction_from_context(conversation_history)
            return FeedbackSignal(
                signal_type=SignalType.IMPLICIT_TOPIC_SHIFT,
                reward=satisfaction,
                confidence=0.4,
                explanation="User changed topic, satisfaction inferred from context"
            )
        
        # Default: neutral
        return FeedbackSignal(
            signal_type=SignalType.IMPLICIT_SILENCE,
            reward=0.0,
            confidence=0.3,
            explanation="Unable to infer user intent"
        )
    
    def _check_keywords(self, msg: str) -> Optional[FeedbackSignal]:
        """Check for explicit positive/negative keywords."""
        msg_lower = msg.lower()
        
        for kw in self.POSITIVE_KEYWORDS:
            if kw in msg_lower:
                return FeedbackSignal(
                    signal_type=SignalType.EXPLICIT_POSITIVE,
                    reward=+1.0,
                    confidence=0.9,
                    explanation=f"Positive keyword detected: '{kw}'"
                )
        
        for kw in self.NEGATIVE_KEYWORDS:
            if kw in msg_lower:
                return FeedbackSignal(
                    signal_type=SignalType.EXPLICIT_NEGATIVE,
                    reward=-1.0,
                    confidence=0.9,
                    explanation=f"Negative keyword detected: '{kw}'"
                )
        
        return None
    
    def _semantic_similarity(self, msg1: str, msg2: str) -> float:
        """Calculate semantic similarity between two messages."""
        if not self.embed:
            return 0.0
        
        emb1 = self.embed(msg1)
        emb2 = self.embed(msg2)
        
        # Cosine similarity
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def _analyze_intent(self, msg: str) -> str:
        """Simple rule-based intent classification."""
        for pattern in self.CLARIFICATION_PATTERNS:
            if pattern in msg:
                return 'clarification'
        
        return 'topic_shift'
    
    def _infer_satisfaction_from_context(self, history: Optional[List[Dict]]) -> float:
        """Infer satisfaction when user changes topic."""
        if not history:
            return 0.0
        
        # If conversation was short, likely dissatisfied or interrupted
        if len(history) < 4:
            return -0.2
        
        # If there were multiple turns without negative signals, likely satisfied
        return 0.3


class DelayedRewardAssigner:
    """
    Assigns rewards to past interactions after conversation completes.
    
    Uses n-step return to distribute final outcome back to earlier steps.
    """
    
    def __init__(self, gamma: float = 0.9, embedding_fn=None):
        self.gamma = gamma  # Discount factor
        self.embed = embedding_fn
    
    def assign_rewards(self, conversation: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        Assign rewards to each assistant turn in the conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            List of (turn, reward) tuples for assistant turns
        """
        final_success = self._judge_conversation_success(conversation)
        n = len(conversation)
        rewards = []
        
        for i, turn in enumerate(conversation):
            if turn.get('role') == 'assistant':
                steps_to_end = n - i
                # Discounted reward based on distance to end
                reward = final_success * (self.gamma ** steps_to_end)
                
                # Blend with immediate feedback if available
                if 'immediate_feedback' in turn:
                    reward = 0.7 * reward + 0.3 * turn['immediate_feedback']
                
                rewards.append((turn, reward))
        
        return rewards
    
    def _judge_conversation_success(self, conversation: List[Dict]) -> float:
        """Judge overall conversation success."""
        if not conversation:
            return 0.0
        
        # Get last user message
        user_msgs = [t for t in conversation if t.get('role') == 'user']
        if not user_msgs:
            return 0.0
        
        last_msg = user_msgs[-1].get('content', '')
        
        # Explicit success signals
        if any(kw in last_msg for kw in ['谢谢', 'ok', '可以', '搞定了', '完美']):
            return +1.0
        
        # Explicit failure signals
        if any(kw in last_msg for kw in ['算了', '放弃', '不问了', '没用']):
            return -1.0
        
        # Check for repetition patterns
        repeat_count = self._count_repeats(conversation)
        if repeat_count >= 3:
            return -0.5
        
        # Default: slightly positive (task completion assumed)
        return 0.3
    
    def _count_repeats(self, conversation: List[Dict]) -> int:
        """Count repeated questions."""
        user_msgs = [t.get('content', '') for t in conversation if t.get('role') == 'user']
        
        repeats = 0
        for i in range(1, len(user_msgs)):
            if self.embed:
                sim = self._calculate_similarity(user_msgs[i-1], user_msgs[i])
                if sim > 0.7:
                    repeats += 1
            else:
                # Simple string overlap as fallback
                if len(set(user_msgs[i-1]) & set(user_msgs[i])) / len(set(user_msgs[i-1])) > 0.8:
                    repeats += 1
        
        return repeats
    
    def _calculate_similarity(self, msg1: str, msg2: str) -> float:
        """Calculate similarity between two messages."""
        if not self.embed:
            return 0.0
        
        emb1 = self.embed(msg1)
        emb2 = self.embed(msg2)
        
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
